"""Единая trust-aware структура находок по лекарственной терапии (Workstream I).

Типы находок (§13.1):
    documentation_gap   - назначение без дозы/кратности/длительности (штрафуемо, trust B)
    protocol_mismatch   - не соответствует режиму протокола (только при trusted протоколе)
    safety_warning      - опасность (дубль НПВП, high-alert без дозы) - штрафуемо
    insufficient_context- доза-зависимо, но нет массы/СКФ/возраста - НЕ штраф (needs_human)
    needs_human         - ненадёжная нормализация ЛС - НЕ штраф

Правила §13.2: не объявлять dose mismatch, если не определено действующее вещество,
нет trustworthy regimen, неизвестны масса/СКФ/возраст при зависимой дозе, источник C/D,
confidence ниже порога.
"""
from __future__ import annotations

from typing import Any

from .kz_evaluation_schema import EvaluationFinding
from .rule_trust import TRUST_B, TRUST_C

_CONF_THRESHOLD = 0.86


def classify_medication_findings(
    case: dict, drug_ctx: dict | None = None,
) -> list[EvaluationFinding]:
    """Вернуть список ``EvaluationFinding`` по назначенной терапии (trust-aware)."""
    treatment = str(case.get("treatment_recommendations") or "").strip()
    findings: list[EvaluationFinding] = []
    if not treatment:
        return findings

    drug_ctx = drug_ctx or {}
    from .medication_parser import active_medication_assignments

    assignments = active_medication_assignments(case)
    active_treatment = "\n".join(
        str(item.get("raw_text") or "") for item in assignments if item.get("raw_text")
    )
    drugs = [
        {
            "surface": item.get("surface") or item.get("drug_name"),
            "inn": item.get("inn"),
            "confidence": item.get("confidence") or 0.0,
            "has_dose": bool(
                item.get("dose_value") is not None
                or item.get("frequency")
                or item.get("schedule")
            ),
        }
        for item in assignments
        if item.get("inn")
    ]
    has_dose = any(bool(d.get("has_dose")) for d in drugs)

    # safety: дубль системных НПВП (не скобки-альтернативы, не гель+таблетка)
    try:
        from .medication_safety import concurrent_systemic_nsaids

        nsaids = concurrent_systemic_nsaids(active_treatment)
    except Exception:  # noqa: BLE001
        nsaids = []
    if len(nsaids) >= 2:
        findings.append(EvaluationFinding(
            code="MED_nsaid_dup", axis="safety", severity="P1", kind="safety_warning",
            passed=False, title_ru="Одновременно ≥2 НПВП",
            detail_ru=", ".join(nsaids[:6]), evidence=active_treatment,
            source_ref="ISMP/клин.практика", trust_level=TRUST_B, penalty_applied=True,
        ))

    # documentation_gap: распознанное ЛС без дозы
    missing_dose = [d for d in drugs if not d.get("has_dose")]
    if missing_dose:
        findings.append(EvaluationFinding(
            code="MED_missing_dose", axis="documentation", severity="P2",
            kind="documentation_gap", passed=False,
            title_ru="Назначение без распознанной дозы/режима",
            detail_ru=", ".join(str(d.get("inn") or "") for d in missing_dose[:6]),
            evidence=active_treatment, source_ref="Пост. №127 / СОП №2",
            trust_level=TRUST_B, penalty_applied=True,
        ))

    # high-alert без дозы -> safety_warning (куратор-база)
    ha = (drug_ctx.get("high_alert") or {}).get("high_alert") if isinstance(drug_ctx.get("high_alert"), dict) else None
    if ha:
        ha_by_inn = {(r.get("inn") or "").lower(): r for r in ha}
        for d in drugs:
            inn = (d.get("inn") or "").lower()
            if inn and inn in ha_by_inn and not d.get("has_dose"):
                findings.append(EvaluationFinding(
                    code="MED_high_alert_no_dose", axis="safety", severity="P1",
                    kind="safety_warning", passed=False,
                    title_ru=f"High-alert препарат без дозы/режима: {d.get('inn')}",
                    evidence=active_treatment, source_ref="ISMP high-alert",
                    trust_level=TRUST_B, penalty_applied=True,
                ))

    # insufficient_context: доза-зависимый препарат, но нет массы/возраста/СКФ -> НЕ штраф
    age = case.get("patient_age_years")
    weight = case.get("patient_weight_kg")
    if drugs and has_dose and (age in (None, "")) and (weight in (None, "")):
        # только помечаем как контекст-дефицит, без штрафа
        findings.append(EvaluationFinding(
            code="MED_dose_context_missing", axis="safety", severity="P3",
            kind="insufficient_context", passed=True,
            title_ru="Проверка дозы ограничена: неизвестны возраст/масса/СКФ",
            detail_ru="Доза-зависимая проверка не выполнялась (нет параметров пациента)",
            trust_level=TRUST_C, penalty_applied=False, needs_human=True,
        ))

    # needs_human: ненадёжная нормализация
    low_conf = [d for d in drugs if 0 < d.get("confidence", 0) < _CONF_THRESHOLD]
    if low_conf:
        findings.append(EvaluationFinding(
            code="MED_unresolved", axis="safety", severity="P3", kind="needs_human",
            passed=True, title_ru="Часть назначений не удалось надёжно нормализовать",
            detail_ru="; ".join(f"{d.get('surface')}→{d.get('inn')}?" for d in low_conf[:6]),
            trust_level=TRUST_C, penalty_applied=False, needs_human=True,
        ))

    return findings
