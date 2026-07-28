"""Единый trust-aware scorer КЗ v3 в shadow-режиме (Workstreams A/C/D ТЗ overnight-v1).

Точка входа:
    evaluate_kz_v3(case, protocol_ctx=None, drug_ctx=None, icd_client=None,
                   legacy=None, mode=None) -> KzEvaluationResultV3

Ключевые архитектурные фиксы относительно legacy (§2.2, §7):
1. Обязательные поля НЕ компенсируются рекомендуемыми (раздельные completion + явные cap).
2. ``None``-блоки не исчезают из знаменателя: считается ``coverage`` (доля оценённого).
3. Недоверенные правила (trust C/D, низкая applicability) переведены в advisory:
   создают ``needs_human`` и снижают coverage/confidence, но НЕ штрафуют и НЕ гейтят.
4. Confidence отделён от клинического score (влияет на статус/gate/ревью, не на балл).
5. Risk-gate: подтверждённый P0 (trust A/B) -> hard cap; C/D critical не гейтит.

Scorer работает в shadow-режиме: не переключает production score и gate.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .kz_evaluation_schema import (
    AxisScores,
    ConfidenceInfo,
    CoverageInfo,
    EvidenceSpan,
    EvaluationFinding,
    EvaluationMode,
    KzEvaluationResultV3,
    Provenance,
    RiskInfo,
    RuleTrustDiagnostics,
)
from .kz_protocol_applicability import assess_applicability
from .rule_trust import TRUST_A, TRUST_B, TRUST_C

_ROOT = Path(__file__).resolve().parent.parent

# Веса осей (renormalize по присутствующим). Совпадают по духу с §4 методологии.
_AXIS_WEIGHTS = {
    "documentation": 0.30,
    "clinical_concordance": 0.35,
    "safety": 0.25,
    "regulatory": 0.10,
}
_SEVERITY_PENALTY = {"P0": 40.0, "P1": 20.0, "P2": 10.0, "P3": 5.0, "ok": 0.0}
_SEVERITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "ok": 9}

# Явные cap документирования (§7.1)
_CAP_NO_DIAGNOSIS = 45.0
_CAP_NO_RECOMMENDATIONS = 55.0
_CAP_NO_OBJECTIVE_PRIMARY = 65.0

_MKB_RE = re.compile(r"^[A-ZА-Я]\d{2}(?:\.\d{1,2})?$", re.I)


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def resolve_mode() -> EvaluationMode:
    """Флаги shadow-режима (§8.3). Дефолты: enabled=1, primary=0, gate=0."""
    return EvaluationMode(
        enabled=_flag("KZ_EVALUATION_V3_ENABLED", "1"),
        primary=_flag("KZ_EVALUATION_V3_PRIMARY", "0"),
        gate=_flag("KZ_EVALUATION_V3_GATE", "0"),
    )


def _txt(case: dict, *keys: str) -> str:
    return " ".join(str(case.get(k) or "").strip() for k in keys if case.get(k)).strip()


def _present(case: dict, *keys: str, minlen: int = 3) -> bool:
    return len(_txt(case, *keys)) >= minlen


_EVIDENCE_FIELDS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "clinical_diagnosis",
    "diagnosis_main_text",
    "mkb_code_main",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "dispensary_info",
)


def _attach_evidence_spans(case: dict, findings: list[EvaluationFinding]) -> None:
    """Связать точную цитату finding с полем и смещениями исходного документа."""
    for finding in findings:
        needle = str(finding.evidence or "").strip()
        if not needle or finding.evidence_span is not None:
            continue
        needle_folded = needle.casefold()
        for field in _EVIDENCE_FIELDS:
            value = str(case.get(field) or "")
            start = value.casefold().find(needle_folded)
            if start < 0:
                continue
            finding.evidence_span = EvidenceSpan(
                field=field,
                start=start,
                end=start + len(needle),
                text=value[start : start + len(needle)],
            )
            break


def _is_primary_visit(case: dict) -> bool:
    blob = _txt(
        case, "raw_text", "complaints", "anamnesis_doctor", "anamnesis_auto",
        "objective_status", "visit_type",
    ).lower()
    if re.search(r"повторн\w*\s+(?:консультац|при[её]м|осмотр)|на\s+контрол", blob):
        return False
    # первичный, если есть жалобы или анамнез
    return _present(case, "complaints") or _present(case, "anamnesis_doctor", "anamnesis_auto")


def _protocol_trust(protocol_ctx: Any) -> str:
    review = str(
        (protocol_ctx.get("review_status") if isinstance(protocol_ctx, dict) else None) or "",
    ).strip().lower()
    if review == "approved":
        return TRUST_A
    if review == "reviewed":
        return TRUST_B
    return TRUST_C  # summary/auto по умолчанию - advisory


# --------------------------------------------------------------------------- #
# Ось A - документирование (coverage-aware, required != optional, §7)
# --------------------------------------------------------------------------- #
def score_documentation(case: dict) -> tuple[float | None, float, list[EvaluationFinding], list[str]]:
    """Вернуть (score, coverage, findings, cap_reasons).

    ``score is None`` -> insufficient_data (пустое/нечитаемое КЗ).
    """
    findings: list[EvaluationFinding] = []
    cap_reasons: list[str] = []

    has_diagnosis = _present(case, "clinical_diagnosis", "diagnosis_main_text")
    has_reco = _present(case, "treatment_recommendations", "exam_recommendations")
    has_objective = _present(case, "objective_status")
    has_complaints = _present(case, "complaints")
    has_anamnesis = _present(case, "anamnesis_doctor", "anamnesis_auto")
    has_exams_reco = _present(case, "exam_recommendations")
    has_follow_up = _present(case, "dispensary_info", "return_date")

    # пустое/нечитаемое КЗ
    any_content = any([
        has_diagnosis, has_reco, has_objective, has_complaints, has_anamnesis,
    ])
    if not any_content:
        return None, 0.0, findings, ["empty_or_unreadable"]

    primary = _is_primary_visit(case)

    # required (обязательные) - НЕ компенсируются рекомендуемыми
    required = {
        "diagnosis": has_diagnosis,
        "recommendations": has_reco,
    }
    if primary:
        required["objective_status"] = has_objective

    # conditional (применимы только на первичном приёме)
    conditional: dict[str, bool] = {}
    if primary:
        conditional["complaints"] = has_complaints

    # recommended - максимум 10% веса
    recommended = {
        "anamnesis": has_anamnesis,
        "exam_recommendations": has_exams_reco,
        "follow_up": has_follow_up,
    }

    def _completion(d: dict[str, bool]) -> float | None:
        if not d:
            return None
        return sum(1 for v in d.values() if v) / len(d)

    req_c = _completion(required)
    cond_c = _completion(conditional)
    rec_c = _completion(recommended)

    # взвешивание с ренормализацией по применимым группам
    parts: list[tuple[float, float]] = []
    if req_c is not None:
        parts.append((req_c, 0.70))
    if cond_c is not None:
        parts.append((cond_c, 0.20))
    if rec_c is not None:
        parts.append((rec_c, 0.10))
    total_w = sum(w for _, w in parts) or 1.0
    score = round(100.0 * sum(v * w for v, w in parts) / total_w, 1)

    # findings по отсутствующим обязательным
    for key, ok in required.items():
        if not ok:
            findings.append(EvaluationFinding(
                code=f"A_missing_{key}", axis="documentation", severity="P2",
                kind="documentation_gap", passed=False,
                title_ru=f"Не заполнен обязательный блок: {key}",
                source_ref="Пост. №127 / СОП №2", trust_level=TRUST_A,
                penalty_applied=True,
            ))
    for key, ok in conditional.items():
        if not ok:
            findings.append(EvaluationFinding(
                code=f"A_missing_{key}", axis="documentation", severity="P3",
                kind="documentation_gap", passed=False,
                title_ru=f"Не заполнен блок первичного приёма: {key}",
                source_ref="Пост. №127", trust_level=TRUST_A, penalty_applied=True,
            ))

    # явные cap (§7.1) - применяются как потолок, не как косвенный штраф
    if not has_diagnosis:
        score = min(score, _CAP_NO_DIAGNOSIS)
        cap_reasons.append(f"нет диагноза -> cap {_CAP_NO_DIAGNOSIS}")
    if not has_reco:
        score = min(score, _CAP_NO_RECOMMENDATIONS)
        cap_reasons.append(f"нет рекомендаций -> cap {_CAP_NO_RECOMMENDATIONS}")
    if primary and not has_objective:
        score = min(score, _CAP_NO_OBJECTIVE_PRIMARY)
        cap_reasons.append(f"нет объективного статуса (первичный) -> cap {_CAP_NO_OBJECTIVE_PRIMARY}")

    # документирование всегда полностью оценимо -> coverage 1.0
    return round(score, 1), 1.0, findings, cap_reasons


# --------------------------------------------------------------------------- #
# Ось B - клиническое соответствие (coverage-aware, trust-aware, §8.1B/§9)
# --------------------------------------------------------------------------- #
def _match_any(needles: list[str], haystack: str) -> bool:
    try:
        from .term_catalog import expand_term
    except Exception:  # noqa: BLE001
        def expand_term(term: str):  # type: ignore
            return (term,)
    hay = haystack.lower()
    for n in needles:
        n = (n or "").strip().lower()
        if not n:
            continue
        for syn in set(expand_term(n)) | {n}:
            s = syn.strip().lower()
            if len(s) >= 3 and s in hay:
                return True
    return False


def _as_text_list(items: Any) -> list[str]:
    out: list[str] = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            out.append(str(it.get("name") or it.get("text") or it.get("title") or ""))
        else:
            for attr in ("name", "text", "title", "drug_name", "drug_group"):
                v = getattr(it, attr, None)
                if v:
                    out.append(str(v))
                    break
    return [x for x in out if x.strip()]


def score_concordance(
    case: dict, protocol_ctx: Any, appl, protocol_trust: str, icd_client=None,
) -> tuple[float | None, float, list[EvaluationFinding]]:
    """Клиническое соответствие. Возвращает (score, coverage, findings).

    Штраф за покрытие протокола применяется ТОЛЬКО если протокол применим
    (appl.penalty_eligible) И правила протокола доверены (trust A/B). Иначе -
    advisory (needs_human), снижает coverage, но не штрафует.
    """
    findings: list[EvaluationFinding] = []
    dx_text = _txt(case, "clinical_diagnosis", "diagnosis_main_text")
    if not dx_text:
        # без диагноза концордантность не оценивается
        return None, 0.0, findings

    complaints = _txt(case, "complaints")
    anamnesis = _txt(case, "anamnesis_doctor", "anamnesis_auto")
    objective = _txt(case, "objective_status", "exam_data")
    exams_done = _txt(case, "exam_recommendations", "exam_data", "manipulations", "service_names")
    treatment = _txt(case, "treatment_recommendations")

    # base subparts (всегда оцениваемы, внутренняя логика trust A)
    scored: list[float] = []  # входит в numerator score (penalty-eligible)
    n_potential = 2  # dx_support + icd_valid всегда потенциально

    # B1 dx support
    has_support = bool(complaints or anamnesis or objective)
    scored.append(100.0 if has_support else 0.0)
    if not has_support:
        findings.append(EvaluationFinding(
            code="B_dx_no_support", axis="clinical_concordance", severity="P1",
            kind="documentation_gap", passed=False,
            title_ru="Диагноз не подкреплён жалобами/анамнезом/осмотром",
            evidence=dx_text, source_ref="§3.4 chain", trust_level=TRUST_A,
            penalty_applied=True,
        ))

    # B2 icd valid form
    code = str(case.get("mkb_code_main") or "").strip()
    icd_ok = bool(code) and bool(_MKB_RE.match(code))
    scored.append(100.0 if icd_ok else 0.0)
    if not icd_ok:
        findings.append(EvaluationFinding(
            code="B_icd_invalid", axis="clinical_concordance", severity="P2",
            kind="documentation_gap", passed=False,
            title_ru="Код МКБ отсутствует или не соответствует формату",
            evidence=code or dx_text, source_ref="МКБ-10", trust_level=TRUST_A,
            penalty_applied=True,
        ))

    protocol_penalty = bool(appl and appl.penalty_eligible and protocol_trust in (TRUST_A, TRUST_B))

    # protocol-dependent subparts
    req_exams = _as_text_list(protocol_ctx.get("required_exams") if isinstance(protocol_ctx, dict) else None)
    tx_groups = _as_text_list(protocol_ctx.get("treatment") if isinstance(protocol_ctx, dict) else None)
    crit = _as_text_list(protocol_ctx.get("diagnostic_criteria") if isinstance(protocol_ctx, dict) else None)

    def _cov(items: list[str], hay: str) -> float:
        covered = sum(1 for e in items if _match_any([e], hay))
        return 100.0 * covered / len(items) if items else 0.0

    for label, items, hay, code_id, title in (
        ("exams", req_exams, exams_done, "B_exams_gap", "обязательные обследования протокола"),
        ("treatment", tx_groups, treatment, "B_tx_offprotocol", "группы лечения протокола"),
        ("criteria", crit, f"{dx_text} {objective} {exams_done}", "B_criteria_absent", "критерии диагноза протокола"),
    ):
        if not items:
            continue
        n_potential += 1
        cov = _cov(items, hay)
        if protocol_penalty:
            scored.append(round(cov, 1))
            if cov < 99:
                missing = [e for e in items if not _match_any([e], hay)]
                findings.append(EvaluationFinding(
                    code=code_id, axis="clinical_concordance",
                    severity="P2" if cov < 50 else "P3",
                    kind="protocol_mismatch", passed=cov >= 99,
                    title_ru=f"Не отражены {title}",
                    detail_ru="; ".join(missing[:8]),
                    source_ref="protocol", trust_level=protocol_trust,
                    penalty_applied=True,
                ))
        else:
            # advisory: не штрафуем, но помечаем needs_human и снижаем coverage
            findings.append(EvaluationFinding(
                code=code_id + "_advisory", axis="clinical_concordance",
                severity="P3", kind="needs_human", passed=True,
                title_ru=f"Покрытие «{title}» не проверено надёжно (нужен методист)",
                detail_ru=(
                    "Протокол не подтверждён достаточным trust level / применимостью - "
                    "требования переведены в advisory (не штрафуются)"
                ),
                source_ref="protocol(advisory)", trust_level=protocol_trust,
                needs_human=True, penalty_applied=False,
            ))

    score = round(sum(scored) / len(scored), 1) if scored else None
    coverage = round(len(scored) / n_potential, 3) if n_potential else 0.0
    return score, coverage, findings


# --------------------------------------------------------------------------- #
# Ось C - безопасность (curated -> penalty; protocol red flags -> trust-aware)
# --------------------------------------------------------------------------- #
_CURATED_SAFETY = {"C_nsaid_dup", "C_ddi", "C_high_alert_no_dose", "C_drug_unresolved"}


def score_safety(
    case: dict, protocol_ctx: Any, drug_ctx: dict | None, appl, protocol_trust: str,
) -> tuple[float | None, float, list[EvaluationFinding]]:
    """Безопасность поверх ``kz_deep_eval._axis_safety`` с trust-переразметкой.

    Куратор-сигналы (НПВП-дубль, DDI, high-alert, STOPP) - trust B, штрафуют/гейтят.
    Red flags протокола - trust протокола; при C/D -> advisory (needs_human).
    """
    try:
        from .kz_deep_eval import _axis_safety
    except Exception:  # noqa: BLE001
        return None, 0.5, []

    _, raw_findings = _axis_safety(case, protocol_ctx, drug_ctx)
    findings: list[EvaluationFinding] = []
    penalty = 0.0
    protocol_penalty = bool(appl and appl.penalty_eligible and protocol_trust in (TRUST_A, TRUST_B))

    for rf in raw_findings:
        code = rf.get("code", "")
        sev = rf.get("severity", "P3")
        passed = bool(rf.get("passed"))
        is_curated = (
            code in _CURATED_SAFETY
            or code.startswith("C_")
            and any(k in code for k in ("stopp", "beers", "start"))
            or code == "C_uncertainty_unrouted"  # из текста КЗ, внутреннее правило
        )
        is_protocol_redflag = code in ("C_red_flag_unrouted",)

        if is_curated:
            trust = TRUST_B
            kind = "safety_warning" if not rf.get("needs_human") else "needs_human"
            do_penalty = not passed and sev != "ok"
        elif is_protocol_redflag:
            trust = protocol_trust
            do_penalty = protocol_penalty and not passed
            kind = "safety_warning" if do_penalty else "needs_human"
        else:
            trust = TRUST_B
            kind = "safety_warning" if not rf.get("needs_human") else "needs_human"
            do_penalty = not passed and sev != "ok" and not rf.get("needs_human")

        if do_penalty:
            penalty += _SEVERITY_PENALTY.get(sev, 5.0)

        findings.append(EvaluationFinding(
            code=code, axis="safety", severity=sev if sev in _SEVERITY_PENALTY else "P3",
            kind=kind, passed=passed,
            title_ru=rf.get("title_ru", ""), detail_ru=rf.get("detail_ru", ""),
            evidence=rf.get("evidence", ""), source_ref=rf.get("source_ref", ""),
            trust_level=trust, penalty_applied=do_penalty,
            needs_human=bool(rf.get("needs_human")) or kind == "needs_human",
        ))

    score = round(max(0.0, 100.0 - penalty), 1)
    # coverage: безопасность оцениваема из текста КЗ + куратор-баз всегда
    has_tx = _present(case, "treatment_recommendations")
    coverage = 1.0 if has_tx else 0.7
    return score, coverage, findings


# --------------------------------------------------------------------------- #
# Ось D - регуляторика (№55, curated -> trust A)
# --------------------------------------------------------------------------- #
def score_regulatory(case: dict) -> tuple[float | None, float, list[EvaluationFinding]]:
    findings: list[EvaluationFinding] = []
    try:
        from .reg55_criteria import evaluate_reg55

        reg55 = evaluate_reg55(case)
    except Exception:  # noqa: BLE001
        return None, 0.0, findings
    score = reg55.get("regulatory_compliance_pct")
    if reg55.get("has_p0_defect"):
        findings.append(EvaluationFinding(
            code="D_reg55_p0", axis="regulatory", severity="P0",
            kind="regulatory_defect", passed=False,
            title_ru="Критический дефект по №55",
            detail_ru="; ".join(
                f.get("title", "") for f in (reg55.get("critical_failed") or [])[:4]
            ),
            source_ref="Пост. №55", trust_level=TRUST_A, penalty_applied=True,
        ))
    coverage = 1.0 if score is not None else 0.0
    return score, coverage, findings


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _build_version() -> str | None:
    p = _ROOT / "rag_server.py"
    try:
        for line in p.read_text(encoding="utf-8").splitlines()[:400]:
            m = re.match(r'\s*BUILD_VERSION\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    except OSError:
        return None
    return None


def _provenance() -> Provenance:
    return Provenance(
        corpus_version="protocol_summaries",
        rules_version="summary+catalog",
        weights_version="kz_evaluation_v3_default",
        build_version=_build_version(),
    )


# --------------------------------------------------------------------------- #
# Risk gate + статус
# --------------------------------------------------------------------------- #
def _worst_trusted(findings: list[EvaluationFinding]) -> tuple[str | None, EvaluationFinding | None]:
    worst_rank = 9
    worst: EvaluationFinding | None = None
    for f in findings:
        if f.passed or not f.penalty_applied:
            continue
        r = _SEVERITY_RANK.get(f.severity, 9)
        if r < worst_rank:
            worst_rank = r
            worst = f
    sev = None if worst is None else worst.severity
    return sev, worst


# --------------------------------------------------------------------------- #
# Главная точка входа
# --------------------------------------------------------------------------- #
def evaluate_kz_v3(
    case: dict,
    *,
    protocol_ctx: Any = None,
    drug_ctx: dict | None = None,
    icd_client=None,
    legacy: dict | None = None,
    mode: EvaluationMode | None = None,
) -> KzEvaluationResultV3:
    mode = mode or resolve_mode()
    appl = assess_applicability(case, protocol_ctx)
    protocol_trust = _protocol_trust(protocol_ctx)

    a_score, a_cov, a_find, a_caps = score_documentation(case)
    if a_score is None and not case_has_any_content(case):
        # пустое КЗ
        result = KzEvaluationResultV3(
            score_pct=None, status="insufficient_data",
            mode=mode, provenance=_provenance(), legacy=dict(legacy or {}),
        )
        return result

    b_score, b_cov, b_find = score_concordance(case, protocol_ctx, appl, protocol_trust, icd_client)
    c_score, c_cov, c_find = score_safety(case, protocol_ctx, drug_ctx, appl, protocol_trust)
    d_score, d_cov, d_find = score_regulatory(case)

    findings = a_find + b_find + c_find + d_find
    _attach_evidence_spans(case, findings)

    axis_vals = {
        "documentation": a_score,
        "clinical_concordance": b_score,
        "safety": c_score,
        "regulatory": d_score,
    }
    axis_cov = {
        "documentation": a_cov if a_score is not None else None,
        "clinical_concordance": b_cov if b_score is not None else None,
        "safety": c_cov if c_score is not None else None,
        "regulatory": d_cov if d_score is not None else None,
    }

    # overall = взвешенное среднее присутствующих осей
    parts = [
        (v, _AXIS_WEIGHTS[k]) for k, v in axis_vals.items() if v is not None
    ]
    overall = round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 1) if parts else None

    # overall coverage - взвешенное среднее покрытий присутствующих осей
    cov_parts = [
        (axis_cov[k], _AXIS_WEIGHTS[k]) for k in axis_vals if axis_cov.get(k) is not None
    ]
    overall_cov = (
        round(sum(v * w for v, w in cov_parts) / sum(w for _, w in cov_parts), 3)
        if cov_parts else 0.0
    )

    # confidence
    doc_parse = 0.9 if a_score is not None and a_score >= 40 else 0.6
    proto_match = appl.applicability_confidence if appl else 0.2
    evidence_match = b_cov if b_score is not None else 0.3
    proto_know = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.3}.get(protocol_trust, 0.2) if appl else 0.2
    conf_vals = [v for v in (doc_parse, proto_match, evidence_match, proto_know) if v is not None]
    conf_overall = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None

    # risk gate
    worst_sev, worst_f = _worst_trusted(findings)
    risk = RiskInfo(worst_severity=worst_sev)
    status = _base_status(overall)

    if worst_sev == "P0":
        capped = min(overall if overall is not None else 40.0, 40.0)
        overall = capped
        status = "critical"
        risk.cap_applied = True
        risk.cap_value = 40.0
        risk.reasons.append(f"подтверждённый P0: {worst_f.title_ru if worst_f else ''}")
    elif worst_sev == "P1":
        capped = min(overall if overall is not None else 60.0, 60.0)
        overall = capped
        risk.cap_applied = True
        risk.cap_value = 60.0
        status = "review" if capped >= 50 else "poor"
        risk.reasons.append(f"подтверждённый P1: {worst_f.title_ru if worst_f else ''}")

    # coverage-aware статус (§7.2) - не понижает ниже risk-статуса
    if status not in ("critical", "poor"):
        if overall_cov < 0.5:
            status = "insufficient_evidence"
        elif overall_cov < 0.8 and status in ("good", "acceptable"):
            status = "limited_evidence"

    # confidence-aware статус (§7.3): низкая уверенность -> требуется ревью
    if conf_overall is not None and conf_overall < 0.5 and status in ("good", "acceptable"):
        status = "review"

    diagnostics = RuleTrustDiagnostics(
        rules_total=len([f for f in findings if not f.passed]),
        rules_penalty_eligible=len([f for f in findings if f.penalty_applied and not f.passed]),
        rules_advisory=len([f for f in findings if f.trust_level == TRUST_C and not f.penalty_applied]),
        rules_heuristic=len([f for f in findings if f.trust_level == "D"]),
    )

    result = KzEvaluationResultV3(
        score_pct=overall,
        status=status,
        axes=AxisScores(**axis_vals),
        coverage=CoverageInfo(
            overall=overall_cov,
            documentation=axis_cov["documentation"],
            clinical_concordance=axis_cov["clinical_concordance"],
            safety=axis_cov["safety"],
            regulatory=axis_cov["regulatory"],
        ),
        confidence=ConfidenceInfo(
            overall=conf_overall,
            document_parse=doc_parse,
            protocol_match=proto_match,
            evidence_match=evidence_match,
            protocol_knowledge=proto_know,
        ),
        risk=risk,
        protocols=[appl] if appl else [],
        findings=findings,
        diagnostics=diagnostics,
        mode=mode,
        provenance=_provenance(),
        legacy=dict(legacy or {}),
    )
    if a_caps:
        result.risk.reasons.extend(a_caps)
    return result


def case_has_any_content(case: dict) -> bool:
    return any(
        _present(case, k) for k in (
            "clinical_diagnosis", "diagnosis_main_text", "complaints",
            "anamnesis_doctor", "anamnesis_auto", "objective_status",
            "treatment_recommendations", "exam_recommendations",
        )
    )


def _base_status(overall: float | None) -> str:
    if overall is None:
        return "insufficient_data"
    if overall >= 80:
        return "good"
    if overall >= 60:
        return "acceptable"
    return "review"


# --------------------------------------------------------------------------- #
# Gate v3 (§14.3) - подготовлен, по умолчанию НЕ включён
# --------------------------------------------------------------------------- #
def gate_v3(result: KzEvaluationResultV3) -> dict[str, Any]:
    """Решение send-gate по v3. Hard block только для подтверждённого P0 (trust A/B).

    C/D findings не блокируют. Низкий score/confidence/coverage -> review, не block.
    """
    block = False
    review = False
    reasons: list[str] = []

    for f in result.findings:
        if f.severity == "P0" and f.penalty_applied and f.trust_level in (TRUST_A, TRUST_B) and not f.passed:
            block = True
            reasons.append(f"confirmed P0 ({f.trust_level}): {f.title_ru}")

    if not block:
        if result.status in ("poor", "critical"):
            review = True
            reasons.append(f"status={result.status}")
        if result.score_pct is not None and result.score_pct < 50:
            review = True
            reasons.append("low score")
        if (result.confidence.overall or 1.0) < 0.5:
            review = True
            reasons.append("low confidence")
        if (result.coverage.overall or 1.0) < 0.5:
            review = True
            reasons.append("low coverage")

    return {
        "block": block,
        "review_required": review or block,
        "reasons": reasons,
        "gate_enabled": result.mode.gate,
    }
