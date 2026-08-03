"""Модель клинического правила протокола 2.0 (ТЗ improve_kz §9).

Legacy-правила из каталога (dict) конвертируются через ``legacy_rule_to_protocol_rule``.
"""
from __future__ import annotations

from typing import Any, Literal

from .consult_schema import SourceRef, _Base

RuleType = Literal[
    "diagnosis_structure_rule",
    "clinical_criterion_rule",
    "diagnostic_criterion_rule",
    "required_exam_rule",
    "conditional_exam_rule",
    "performed_or_recommended_exam_rule",
    "treatment_group_rule",
    "drug_rule",
    "drug_dose_rule",
    "drug_duration_rule",
    "non_drug_rule",
    "follow_up_rule",
    "routing_rule",
    "red_flag_rule",
    "contraindication_rule",
    "age_applicability_rule",
    "sex_applicability_rule",
    "pregnancy_applicability_rule",
    "informational_rule",
    # legacy aliases
    "diagnosis_formula",
    "diagnostic_criterion",
    "required_exam",
    "keyword_presence",
    "population_mismatch",
]

RuleSeverity = Literal[
    "required", "conditional", "recommended", "warning", "forbidden", "informational",
]

RuleSource = Literal["legacy", "summary", "manual", "table", "llm_draft"]


class RuleApplicability(_Base):
    age_groups: list[str] = []
    age_min_years: int | None = None
    age_max_years: int | None = None
    sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"
    condition_certainty: list[Literal["confirmed", "suspected", "unclear"]] = []
    care_setting: list[str] = []


class ProtocolRule(_Base):
    rule_id: str
    protocol_id: str = ""
    condition_id: str | None = None
    condition_name: str | None = None
    icd10_codes: list[str] = []

    rule_type: RuleType = "informational_rule"
    severity: RuleSeverity = "recommended"
    rule_source: RuleSource = "legacy"
    generated_from_summary: bool = False
    summary_id: str | None = None
    summary_version: str | None = None

    applicability: RuleApplicability = RuleApplicability()
    evidence_targets: list[str] = []
    criteria: list[dict[str, Any]] = []
    expected_items: list[str] = []
    forbidden_items: list[str] = []

    source: SourceRef = SourceRef()
    confidence: float = 1.0  # уверенность извлечения правила (0-1)


_LEGACY_TYPE_MAP: dict[str, RuleType] = {
    "diagnosis_formula": "diagnosis_structure_rule",
    "diagnostic_criterion": "diagnostic_criterion_rule",
    "required_exam": "required_exam_rule",
    "keyword_presence": "drug_rule",
    "population_mismatch": "age_applicability_rule",
}


def legacy_rule_to_protocol_rule(rule: dict[str, Any]) -> ProtocolRule:
    """Конвертирует dict из каталога rules в ProtocolRule без потери source."""
    rt = str(rule.get("rule_type") or "informational_rule")
    mapped = _LEGACY_TYPE_MAP.get(rt, rt)  # type: ignore[arg-type]
    src = rule.get("source") or {}
    source = SourceRef(
        local_path=src.get("source_path"),
        protocol_id=str(src.get("protocol_id") or "") or None,
        section_title=src.get("section_title"),
        quote=(src.get("quote") or "")[:400] or None,
        page_start=src.get("page"),
    )
    expected: list[str] = []
    if rule.get("exam"):
        expected.append(str(rule["exam"]))
    if rule.get("keyword"):
        expected.append(str(rule["keyword"]))
    req_components = list(rule.get("required_components") or [])
    if req_components and all(isinstance(x, str) for x in req_components):
        expected.extend(str(x) for x in req_components)
    raw_criteria = list(rule.get("criteria") or [])
    criteria = [c for c in raw_criteria if isinstance(c, dict)]
    if not criteria and raw_criteria and not req_components:
        criteria = [c for c in raw_criteria if isinstance(c, dict)]
    targets: list[str] = []
    if mapped in ("required_exam_rule", "conditional_exam_rule", "performed_or_recommended_exam_rule"):
        targets = ["performed_exams", "recommended_exams"]
    elif mapped in ("drug_rule", "drug_dose_rule", "drug_duration_rule", "treatment_group_rule"):
        targets = ["medications", "treatment"]
    elif mapped == "diagnosis_structure_rule":
        targets = ["diagnosis"]
    elif mapped == "diagnostic_criterion_rule":
        targets = ["complaints", "anamnesis", "objective_status"]
    sev = str(rule.get("severity") or "recommended")
    if sev in ("critical", "high"):
        sev_norm = "required"
    elif sev == "warning":
        sev_norm = "warning"
    elif sev == "info":
        sev_norm = "informational"
    else:
        sev_norm = "conditional"
    return ProtocolRule(
        rule_id=str(rule.get("rule_id") or ""),
        protocol_id=str(src.get("protocol_id") or ""),
        condition_id=str(rule.get("condition_id") or "") or None,
        rule_type=mapped,  # type: ignore[arg-type]
        severity=sev_norm,  # type: ignore[arg-type]
        evidence_targets=targets,
        expected_items=expected,
        criteria=criteria,
        source=source,
        confidence=0.85 if rt != "keyword_presence" else 0.6,
        rule_source=str(rule.get("rule_source") or "legacy"),  # type: ignore[arg-type]
    )


def rule_applicable_to_patient(
    rule: ProtocolRule,
    patient: dict[str, Any],
    *,
    diagnosis_certainty: str | None = None,
) -> bool:
    """Правило неприменимо - не должно снижать score."""
    appl = rule.applicability
    if appl.condition_certainty and diagnosis_certainty:
        if diagnosis_certainty not in appl.condition_certainty:
            return False
    sex = (patient.get("sex") or "unknown").lower()
    if appl.sex not in ("unknown", "any") and sex in ("male", "female") and appl.sex != sex:
        return False
    preg = patient.get("pregnancy")
    if appl.pregnancy == "required" and preg is not True:
        return False
    if appl.pregnancy == "excluded" and preg is True:
        return False
    age = patient.get("age_years")
    if age is not None:
        if appl.age_min_years is not None and age < appl.age_min_years:
            return False
        if appl.age_max_years is not None and age > appl.age_max_years:
            return False
    aud = (patient.get("adult_or_child") or "").lower()
    if appl.age_groups and aud and aud not in appl.age_groups and "any" not in appl.age_groups:
        return False
    return True
