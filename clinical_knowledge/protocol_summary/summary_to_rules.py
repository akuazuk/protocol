"""Генерация ProtocolRule из Protocol Summary."""
from __future__ import annotations

import re
from typing import Any

from ..rule_model import ProtocolRule, RuleApplicability
from .schema import (
    ConditionSummary,
    CriteriaBlock,
    CriterionItem,
    ExamRequirement,
    FollowUpRequirement,
    ProtocolApplicability,
    ProtocolSummary,
    RedFlagItem,
    SummarySourceRef,
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (s or "").lower()).strip("_")[:48]


def _applicability_from_summary(appl: ProtocolApplicability | None) -> RuleApplicability:
    if not appl:
        return RuleApplicability()
    age_groups = [p for p in appl.population if p not in ("unknown", "pregnant", "adult_and_child")]
    if "adult_and_child" in appl.population:
        age_groups.extend(["adult", "child"])
    pregnancy = appl.pregnancy
    sex = appl.sex
    if "pregnant" in appl.population:
        pregnancy = "required"
        if sex == "unknown":
            sex = "female"
        if "adult" not in age_groups:
            age_groups.append("adult")
    return RuleApplicability(
        age_groups=age_groups,
        age_min_years=appl.age_min_years,
        age_max_years=appl.age_max_years,
        sex=sex,
        pregnancy=pregnancy,
        care_setting=[c for c in appl.care_setting if c != "unknown"],
    )


def _merge_applicability(
    protocol: ProtocolApplicability,
    condition: ProtocolApplicability | None,
) -> RuleApplicability:
    base = _applicability_from_summary(protocol)
    if not condition:
        return base
    over = _applicability_from_summary(condition)
    return RuleApplicability(
        age_groups=over.age_groups or base.age_groups,
        age_min_years=over.age_min_years if over.age_min_years is not None else base.age_min_years,
        age_max_years=over.age_max_years if over.age_max_years is not None else base.age_max_years,
        sex=over.sex if over.sex != "unknown" else base.sex,
        pregnancy=over.pregnancy if over.pregnancy != "unknown" else base.pregnancy,
        care_setting=over.care_setting or base.care_setting,
    )


def _rule_base(
    summary: ProtocolSummary,
    cond: ConditionSummary,
    suffix: str,
    rule_type: str,
    *,
    severity: str = "recommended",
    expected: list[str] | None = None,
    targets: list[str] | None = None,
    source_ref: SummarySourceRef | None = None,
    criteria: list[dict[str, Any]] | None = None,
) -> ProtocolRule:
    rid = f"{summary.protocol_id}__{cond.condition_id}__{suffix}"
    src = (source_ref or SummarySourceRef(protocol_id=summary.protocol_id)).to_source_ref()
    return ProtocolRule(
        rule_id=rid,
        protocol_id=summary.protocol_id,
        condition_id=cond.condition_id,
        condition_name=cond.name,
        icd10_codes=list(cond.icd10_codes),
        rule_type=rule_type,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        applicability=_merge_applicability(summary.applicability, cond.condition_applicability),
        evidence_targets=targets or [],
        expected_items=expected or [],
        criteria=criteria or [],
        source=src,
        confidence=0.92,
        rule_source="summary",
        generated_from_summary=True,
        summary_id=summary.protocol_id,
        summary_version=summary.summary_version,
    )


def _criteria_rules(
    summary: ProtocolSummary,
    cond: ConditionSummary,
    block: CriteriaBlock | None,
    rule_type: str,
    prefix: str,
) -> list[ProtocolRule]:
    if not block:
        return []
    rules: list[ProtocolRule] = []
    for i, item in enumerate(block.required):
        rules.append(_criterion_rule(summary, cond, item, rule_type, f"{prefix}_req_{i}", severity="required"))
    for i, item in enumerate(block.optional):
        rules.append(_criterion_rule(summary, cond, item, rule_type, f"{prefix}_opt_{i}", severity="recommended"))
    return rules


def _criterion_rule(
    summary: ProtocolSummary,
    cond: ConditionSummary,
    item: CriterionItem,
    rule_type: str,
    suffix: str,
    *,
    severity: str,
) -> ProtocolRule:
    targets = list(item.evidence_targets) or ["complaints", "anamnesis"]
    return _rule_base(
        summary, cond, suffix, rule_type,
        severity=severity,
        expected=[item.text],
        targets=targets,
        source_ref=item.source_ref,
        criteria=[{"text": item.text, "operator": item.operator, "values": item.values}],
    )


def _exam_rules(
    summary: ProtocolSummary,
    cond: ConditionSummary,
    exams: list[ExamRequirement],
    rule_type: str,
    prefix: str,
) -> list[ProtocolRule]:
    rules: list[ProtocolRule] = []
    for i, exam in enumerate(exams):
        sev = "required" if exam.requirement_level == "required" else "conditional"
        rules.append(
            _rule_base(
                summary, cond, f"{prefix}_{i}", rule_type,
                severity=sev,
                expected=[exam.name] + exam.aliases[:2],
                targets=["performed_exams", "recommended_exams"],
                source_ref=exam.source_ref,
                criteria=[{"requirement_level": exam.requirement_level, "required_if": exam.required_if}],
            ),
        )
    return rules


def _red_flag_rules(
    summary: ProtocolSummary,
    cond: ConditionSummary,
    flags: list[RedFlagItem],
    prefix: str,
) -> list[ProtocolRule]:
    rules: list[ProtocolRule] = []
    for i, rf in enumerate(flags):
        sev = "required" if rf.severity in ("high", "critical") else "warning"
        rules.append(
            _rule_base(
                summary, cond, f"{prefix}_{i}", "red_flag_rule",
                severity=sev,
                expected=[rf.text] + rf.aliases[:3],
                targets=["diagnosis", "complaints", "routing", "follow_up"],
                source_ref=rf.source_ref,
                criteria=[
                    {
                        "red_flag_type": rf.red_flag_type,
                        "severity": rf.severity,
                        "expected_actions": rf.expected_actions,
                        "cap_if_unhandled": rf.cap_if_unhandled,
                    },
                ],
            ),
        )
    return rules


def condition_to_protocol_rules(summary: ProtocolSummary, cond: ConditionSummary) -> list[ProtocolRule]:
    rules: list[ProtocolRule] = []
    if cond.diagnosis_structure:
        for i, comp in enumerate(cond.diagnosis_structure.required_components):
            rules.append(
                _rule_base(
                    summary, cond, f"dx_struct_{i}", "diagnosis_structure_rule",
                    severity="required" if comp.required else "recommended",
                    expected=[comp.name],
                    targets=["diagnosis"],
                    source_ref=comp.source_ref,
                ),
            )
    rules.extend(_criteria_rules(summary, cond, cond.clinical_criteria, "clinical_criterion_rule", "clinical"))
    rules.extend(_criteria_rules(summary, cond, cond.diagnostic_criteria, "diagnostic_criterion_rule", "diagnostic"))
    rules.extend(_exam_rules(summary, cond, cond.required_exams, "required_exam_rule", "req_exam"))
    rules.extend(_exam_rules(summary, cond, cond.conditional_exams, "conditional_exam_rule", "cond_exam"))

    if cond.treatment:
        for i, g in enumerate(cond.treatment.drug_groups):
            rules.append(
                _rule_base(
                    summary, cond, f"drug_grp_{i}", "treatment_group_rule",
                    expected=[g.drug_group],
                    targets=["treatment", "medications"],
                    source_ref=g.source_ref,
                    criteria=[{"indication": g.indication}],
                ),
            )
        for i, d in enumerate(cond.treatment.drugs):
            label = d.drug_name or d.active_substance or d.drug_group or "препарат"
            rt = "drug_rule"
            if d.dose_text:
                rt = "drug_dose_rule"
            if d.duration_text:
                rt = "drug_duration_rule"
            rules.append(
                _rule_base(
                    summary, cond, f"drug_{i}", rt,
                    expected=[label],
                    targets=["medications", "treatment"],
                    source_ref=d.source_ref,
                    criteria=[
                        {
                            "dose_text": d.dose_text,
                            "frequency_text": d.frequency_text,
                            "duration_text": d.duration_text,
                        },
                    ],
                ),
            )
        for i, nd in enumerate(cond.treatment.non_drug):
            rules.append(
                _rule_base(
                    summary, cond, f"non_drug_{i}", "non_drug_rule",
                    expected=[nd.text],
                    targets=["treatment"],
                    source_ref=nd.source_ref,
                ),
            )

    for i, fu in enumerate(cond.follow_up):
        rules.append(
            _rule_base(
                summary, cond, f"follow_up_{i}", "follow_up_rule",
                expected=[fu.text],
                targets=["follow_up"],
                source_ref=fu.source_ref,
                criteria=[{"timing": fu.timing, "expected_actions": fu.expected_actions}],
            ),
        )

    for i, route in enumerate(cond.routing):
        rules.append(
            _rule_base(
                summary, cond, f"routing_{i}", "routing_rule",
                expected=[route.text],
                targets=["routing"],
                source_ref=route.source_ref,
            ),
        )

    rules.extend(_red_flag_rules(summary, cond, cond.red_flags, "red_flag"))
    return rules


def summary_to_protocol_rules(summary: ProtocolSummary) -> list[ProtocolRule]:
    rules: list[ProtocolRule] = []
    for cond in summary.conditions:
        rules.extend(condition_to_protocol_rules(summary, cond))
    for i, rf in enumerate(summary.global_red_flags):
        dummy = ConditionSummary(condition_id="_global", name="Global", icd10_codes=[])
        rules.extend(_red_flag_rules(summary, dummy, [rf], f"global_rf_{i}"))
    return rules


_LEGACY_TYPE_MAP: dict[str, str] = {
    "diagnosis_structure_rule": "diagnosis_formula",
    "clinical_criterion_rule": "diagnostic_criterion",
    "diagnostic_criterion_rule": "diagnostic_criterion",
    "required_exam_rule": "required_exam",
    "conditional_exam_rule": "required_exam",
    "performed_or_recommended_exam_rule": "required_exam",
    "treatment_group_rule": "keyword_presence",
    "drug_rule": "keyword_presence",
    "drug_dose_rule": "keyword_presence",
    "drug_duration_rule": "keyword_presence",
    "non_drug_rule": "keyword_presence",
    "follow_up_rule": "keyword_presence",
    "routing_rule": "keyword_presence",
    "red_flag_rule": "red_flag_rule",
    "contraindication_rule": "keyword_presence",
}


def protocol_rule_to_legacy_dict(rule: ProtocolRule) -> dict[str, Any]:
    """Конвертирует ProtocolRule (summary) в dict для rule_checker."""
    legacy_type = _LEGACY_TYPE_MAP.get(str(rule.rule_type), str(rule.rule_type))
    src = rule.source
    source = {
        "protocol_id": rule.protocol_id or src.protocol_id,
        "source_path": src.local_path,
        "section_title": src.section_title,
        "quote": src.quote,
        "page": src.page_start,
    }
    out: dict[str, Any] = {
        "rule_id": rule.rule_id,
        "rule_type": legacy_type,
        "severity": rule.severity,
        "condition_id": rule.condition_id,
        "source": source,
        "rule_source": rule.rule_source,
        "generated_from_summary": rule.generated_from_summary,
        "summary_id": rule.summary_id,
        "criteria": rule.criteria,
    }
    if legacy_type == "diagnosis_formula":
        out["required_components"] = list(rule.expected_items)
    elif legacy_type == "required_exam":
        out["exam"] = rule.expected_items[0] if rule.expected_items else ""
    elif legacy_type == "diagnostic_criterion":
        out["logic"] = "any_of"
        if not out["criteria"]:
            out["criteria"] = [{"finding": t} for t in rule.expected_items]
        out["description_ru"] = rule.expected_items[0] if rule.expected_items else ""
    elif legacy_type == "red_flag_rule":
        crit = rule.criteria[0] if rule.criteria else {}
        out["keywords"] = list(rule.expected_items)
        out["red_flag_type"] = crit.get("red_flag_type")
        out["cap_if_unhandled"] = crit.get("cap_if_unhandled")
    else:
        out["keyword"] = rule.expected_items[0] if rule.expected_items else ""
        out["message_ru"] = rule.expected_items[0] if rule.expected_items else ""
    return out
