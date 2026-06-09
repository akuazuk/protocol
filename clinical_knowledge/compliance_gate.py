"""Решение о допуске КЗ к подписи и отправке в государственный контур (для МИС).

Protocol не подписывает и не импортирует пакеты в ЦИСЗ – возвращает рекомендацию
«можно / нельзя / нужно подтверждение врача», которую МИС «Айболит» применяет
по локальной политике учреждения.
"""
from __future__ import annotations

import os
from typing import Any, Literal

from .consult_schema import ComplianceReport

GateMode = Literal["inform", "soft_gate", "hard_gate", "critical_only"]
SendRiskLevel = Literal["low", "medium", "high", "blocked"]


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_mode() -> GateMode:
    raw = (os.environ.get("COMPLIANCE_GATE_MODE") or "inform").strip().lower()
    if raw in ("inform", "soft_gate", "hard_gate", "critical_only"):
        return raw  # type: ignore[return-value]
    return "inform"


def _has_blocking_critical(report: ComplianceReport) -> bool:
    if report.overall_status == "manual_review_required":
        return True
    for iss in report.critical_issues:
        sev = (iss.severity or "").lower()
        if sev in ("critical", "high"):
            return True
    for s in report.safety_assessments:
        if s.severity == "critical" and s.status not in ("handled", "partially_handled"):
            return True
    return False


def resolve_gate_score(
    report: ComplianceReport,
    *,
    headline_score: float | None = None,
) -> float | None:
    """Единый балл для gate: min(гибридный headline, структурный), если оба заданы."""
    structural = report.overall_score
    if headline_score is not None and structural is not None:
        return min(float(headline_score), float(structural))
    if headline_score is not None:
        return float(headline_score)
    return structural


def evaluate_send_gate(
    report: ComplianceReport,
    *,
    headline_score: float | None = None,
    mode: GateMode | None = None,
    min_score_hard: float | None = None,
    min_score_soft: float | None = None,
) -> dict[str, Any]:
    """Возвращает политику допуска к ЭЦП/отправке для интеграции с МИС."""
    mode = mode or _env_mode()
    hard_thr = min_score_hard if min_score_hard is not None else _env_float("COMPLIANCE_GATE_MIN_SCORE", 70.0)
    soft_thr = min_score_soft if min_score_soft is not None else _env_float("COMPLIANCE_GATE_SOFT_MIN_SCORE", 55.0)

    structural_score = report.overall_score
    score = resolve_gate_score(report, headline_score=headline_score)
    critical_block = _has_blocking_critical(report)
    confidence_low = (report.confidence_score or 100) < 50

    requires_override = False
    gate_allowed = True
    block_reason: str | None = None
    send_risk: SendRiskLevel = "low"

    if critical_block:
        send_risk = "high"
    elif score is not None and score < soft_thr:
        send_risk = "medium"
    elif score is not None and score < hard_thr:
        send_risk = "medium"

    if mode == "inform":
        gate_allowed = True
        if critical_block:
            block_reason = "Критические замечания – рекомендуется исправить до подписи (режим inform: блокировка отключена)."
        elif score is not None and score < hard_thr:
            block_reason = f"Оценка {score:.0f}% ниже рекомендуемого порога {hard_thr:.0f}% – доработайте КЗ перед отправкой."

    elif mode == "critical_only":
        gate_allowed = not critical_block
        if not gate_allowed:
            send_risk = "blocked"
            block_reason = "Критические замечания по безопасности или структуре – подпись заблокирована политикой учреждения."

    elif mode == "hard_gate":
        low_score = score is not None and score < hard_thr
        gate_allowed = not critical_block and not low_score and not confidence_low
        if not gate_allowed:
            send_risk = "blocked"
            if critical_block:
                block_reason = "Критические замечания – подпись заблокирована."
            elif low_score:
                block_reason = f"Оценка {score:.0f}% ниже порога {hard_thr:.0f}% – подпись заблокирована."
            elif confidence_low:
                block_reason = "Низкая уверенность разбора – подпись заблокирована; проверьте текст КЗ вручную."

    elif mode == "soft_gate":
        gate_allowed = not critical_block
        low_score = score is not None and score < hard_thr
        if critical_block:
            send_risk = "blocked"
            block_reason = "Критические замечания – подпись заблокирована."
            gate_allowed = False
        elif low_score or confidence_low:
            requires_override = True
            if low_score:
                block_reason = (
                    f"Оценка {score:.0f}% ниже {hard_thr:.0f}% – требуется подтверждение врача «Подписать всё равно»."
                )
            else:
                block_reason = "Низкая уверенность разбора – требуется подтверждение врача."

    return {
        "gate_mode": mode,
        "gate_allowed": gate_allowed,
        "requires_override": requires_override,
        "block_reason_ru": block_reason,
        "send_risk_level": send_risk,
        "min_score_hard": hard_thr,
        "min_score_soft": soft_thr,
        "gate_score": score,
        "headline_score": headline_score,
        "structural_score": structural_score,
        "overall_score": score,
        "overall_status": report.overall_status,
        "critical_issues_count": len(report.critical_issues),
        "confidence_score": report.confidence_score,
        "disclaimer_ru": (
            "Ориентир методслужбы по клиническим протоколам Минздрава РБ; "
            "не заменяет МЭЭ и не является юридическим заключением."
        ),
    }


def evaluate_send_gate_from_compliance(
    compliance: dict[str, Any],
    *,
    headline_score: float | None = None,
    mode: GateMode | None = None,
) -> dict[str, Any]:
    """Gate по JSON compliance (после гибридного headline в пайплайне)."""
    from .consult_schema import ComplianceIssue, SafetyAssessment

    issues = [ComplianceIssue.model_validate(i) for i in (compliance.get("critical_issues") or [])]
    safety = [SafetyAssessment.model_validate(s) for s in (compliance.get("safety_assessments") or [])]
    report = ComplianceReport(
        consultation_id=str(compliance.get("consultation_id") or ""),
        overall_score=compliance.get("overall_score"),
        overall_status=str(compliance.get("overall_status") or "insufficient_data"),
        confidence_score=compliance.get("confidence_score"),
        critical_issues=issues,
        safety_assessments=safety,
    )
    return evaluate_send_gate(report, headline_score=headline_score, mode=mode)
