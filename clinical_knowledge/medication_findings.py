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

import re
from typing import Any

from .kz_evaluation_schema import EvaluationFinding
from .rule_trust import TRUST_B, TRUST_C

_DOSE_RE = re.compile(r"\d+\s*(мг|mg|мкг|ме|ед|мл|г\b|таб|капс)", re.I)
_CONF_THRESHOLD = 0.86


def _drugs(text: str) -> list[dict[str, Any]]:
    try:
        from .drug_normalizer import extract_drugs

        return extract_drugs(text) or []
    except Exception:  # noqa: BLE001
        return []


def classify_medication_findings(
    case: dict, drug_ctx: dict | None = None,
) -> list[EvaluationFinding]:
    """Вернуть список ``EvaluationFinding`` по назначенной терапии (trust-aware)."""
    treatment = str(case.get("treatment_recommendations") or "").strip()
    findings: list[EvaluationFinding] = []
    if not treatment:
        return findings

    drug_ctx = drug_ctx or {}
    drugs = _drugs(treatment)
    has_dose = bool(_DOSE_RE.search(treatment))

    # safety: дубль НПВП (детерминированный куратор-сигнал)
    try:
        from .medication_safety import nsaid_labels_in_text

        nsaids = set(nsaid_labels_in_text(treatment))
    except Exception:  # noqa: BLE001
        nsaids = set()
    if len(nsaids) >= 2:
        findings.append(EvaluationFinding(
            code="MED_nsaid_dup", axis="safety", severity="P1", kind="safety_warning",
            passed=False, title_ru="Одновременно ≥2 НПВП",
            detail_ru=", ".join(sorted(nsaids)[:6]), evidence=treatment,
            source_ref="ISMP/клин.практика", trust_level=TRUST_B, penalty_applied=True,
        ))

    # documentation_gap: распознанное ЛС без дозы
    if drugs and not has_dose:
        findings.append(EvaluationFinding(
            code="MED_missing_dose", axis="documentation", severity="P2",
            kind="documentation_gap", passed=False,
            title_ru="Назначение без распознанной дозы/режима",
            evidence=treatment, source_ref="Пост. №127 / СОП №2",
            trust_level=TRUST_B, penalty_applied=True,
        ))

    # high-alert без дозы -> safety_warning (куратор-база)
    ha = (drug_ctx.get("high_alert") or {}).get("high_alert") if isinstance(drug_ctx.get("high_alert"), dict) else None
    if ha:
        ha_by_inn = {(r.get("inn") or "").lower(): r for r in ha}
        for d in drugs:
            inn = (d.get("inn") or "").lower()
            if inn and inn in ha_by_inn and not has_dose:
                findings.append(EvaluationFinding(
                    code="MED_high_alert_no_dose", axis="safety", severity="P1",
                    kind="safety_warning", passed=False,
                    title_ru=f"High-alert препарат без дозы/режима: {d.get('inn')}",
                    evidence=treatment, source_ref="ISMP high-alert",
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
