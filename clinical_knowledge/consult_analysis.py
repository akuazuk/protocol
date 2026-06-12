"""Высокоуровневый оркестратор структурного анализа КЗ.

Связывает: парсинг -> подбор протоколов с применимостью -> детерминированные правила
-> compliance engine -> отчёты. Единая точка входа для интеграции в пайплайн/CLI.
"""
from __future__ import annotations

from typing import Any

from .compliance_engine import build_compliance_report
from .condition_registry import infer_conditions_hints
from .consult_parser import parse_consultation
from .consult_report import report_to_html, report_to_json, report_to_markdown
from .consult_schema import ComplianceIssue, ConsultationDocument, SourceRef
from .protocol_match import annotate_applicability, match_protocol_cards, match_protocol_cards_for_diagnoses
from .rubric_extractors import (
    extract_rubric_specifics,
    normalize_rubric_slug,
    rubric_from_icd,
    rubric_slugs_from_matches,
    specialty_to_rubric,
)
from .rule_checker import collect_catalog_rules, run_rule_checker


def facts_from_document(doc: ConsultationDocument) -> dict[str, Any]:
    """Преобразует ConsultationDocument в legacy-facts dict для matcher/rule_checker."""
    icd = [d.icd10_code for d in doc.diagnoses if d.icd10_code]
    diag_text = doc.sections.diagnosis_text or "; ".join(d.raw_text for d in doc.diagnoses)
    complaints = []
    if doc.sections.complaints:
        complaints = [c.strip() for c in doc.sections.complaints.split("\n") if c.strip()][:5]
    low = (doc.raw_text or "").lower()
    hints = infer_conditions_hints(low, icd)
    return {
        "patient_context": {
            "age_years": doc.patient.age_years,
            "sex": doc.patient.sex if doc.patient.sex in ("male", "female") else None,
            "adult_or_child": doc.patient.adult_or_child if doc.patient.adult_or_child != "unknown" else None,
            "pregnancy": doc.patient.pregnancy,
        },
        "consultation": {
            "complaints": complaints,
            "diagnosis_text": diag_text,
            "icd10": icd,
            "conditions_hint": hints,
            "text_sample": (doc.raw_text or "")[:2000],
        },
        "extraction_method": "structured_parser",
    }


def _patient_dict(doc: ConsultationDocument) -> dict[str, Any]:
    return {
        "age_years": doc.patient.age_years,
        "adult_or_child": doc.patient.adult_or_child,
        "sex": doc.patient.sex,
        "pregnancy": doc.patient.pregnancy,
    }


def _summary_source_refs(summaries: list[Any], extra_rules: list[dict[str, Any]]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[str] = set()
    for s in summaries:
        lp = getattr(s.source, "local_path", None) if hasattr(s, "source") else None
        pid = getattr(s, "protocol_id", None)
        key = f"{pid}:{lp}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(SourceRef(local_path=lp, protocol_id=pid, section_type="summary_card"))
    for r in extra_rules:
        if r.get("rule_source") != "summary":
            continue
        src = r.get("source") or {}
        q = (src.get("quote") or "")[:400] or None
        key = f"{src.get('protocol_id')}:{src.get('page')}:{q}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            SourceRef(
                protocol_id=src.get("protocol_id"),
                local_path=src.get("source_path") or src.get("local_path"),
                page_start=src.get("page") or src.get("page_start"),
                section_title=src.get("section_title"),
                quote=q,
                section_type="summary_rule",
            ),
        )
    return refs


def _legacy_source_refs(matches: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[str] = set()
    for m in matches[:8]:
        p = m.get("source_path") or m.get("local_path")
        pid = str(m.get("protocol_id") or m.get("card_id") or "")
        key = f"{pid}:{p}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(SourceRef(local_path=p, protocol_id=pid or None, section_type="protocol_card"))
    for f in findings:
        if f.get("rule_source") == "summary":
            continue
        src = f.get("source") or {}
        if not isinstance(src, dict):
            continue
        p = src.get("source_path") or src.get("local_path")
        if not p:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            SourceRef(
                local_path=p,
                protocol_id=src.get("protocol_id"),
                page_start=src.get("page") or src.get("page_start"),
                section_title=src.get("section_title"),
                quote=(src.get("quote") or "")[:400] or None,
                section_type="legacy_rule",
            ),
        )
    return refs


def _build_method_comparison(
    legacy_comp: dict[str, Any],
    current_comp: dict[str, Any],
) -> dict[str, Any]:
    from .protocol_summary.summary_compare import compliance_metrics

    lm = compliance_metrics(legacy_comp)
    cm = compliance_metrics(current_comp)
    return {
        "legacy": lm,
        "current": cm,
        "same_decision": lm.get("overall_score") == cm.get("overall_score"),
        "score_delta": (
            (cm.get("overall_score") or 0) - (lm.get("overall_score") or 0)
            if cm.get("overall_score") is not None and lm.get("overall_score") is not None
            else None
        ),
        "critical_issue_delta": (cm.get("critical_issues") or 0) - (lm.get("critical_issues") or 0),
        "summary_rules_in_evidence": sum(
            1 for e in (current_comp.get("evidence_map") or [])
            if e.get("rule_source") == "summary"
        ),
    }


def analyze_consultation_text(
    raw_text: str,
    *,
    consultation_id: str = "consult",
    source_file: str = "",
    source_file_type: str = "",
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
    match_limit: int = 8,
    with_markdown: bool = True,
    analysis_mode: str | None = None,
) -> dict[str, Any]:
    """Полный структурный разбор + оценка соответствия для одного КЗ."""
    doc = parse_consultation(
        raw_text,
        consultation_id=consultation_id,
        source_file=source_file,
        source_file_type=source_file_type,
        demographics_meta=demographics_meta,
    )
    facts = facts_from_document(doc)

    doctor_rubric = specialty_to_rubric(doc.doctor_specialty)
    icd_rubric = rubric_from_icd([d.icd10_code for d in doc.diagnoses if d.icd10_code])
    effective_slug = specialty_slug or doctor_rubric or icd_rubric

    try:
        dx_payload = [
            {
                "diagnosis_id": d.diagnosis_id or f"dx{i}",
                "icd10_code": d.icd10_code,
                "raw_text": d.raw_text,
                "certainty": d.certainty,
            }
            for i, d in enumerate(doc.diagnoses)
        ]
        applicable, not_applicable = match_protocol_cards_for_diagnoses(
            facts,
            dx_payload,
            specialty_slug=effective_slug,
            limit_per_dx=3,
            limit_total=match_limit,
        )
        matches = applicable
        if not matches and effective_slug:
            applicable, not_applicable = match_protocol_cards_for_diagnoses(
                facts, dx_payload, specialty_slug=None, limit_per_dx=3, limit_total=match_limit,
            )
            matches = applicable
    except Exception:
        matches = []
        not_applicable = []
        try:
            matches = match_protocol_cards(facts, specialty_slug=effective_slug, limit=match_limit)
            if not matches and effective_slug:
                matches = match_protocol_cards(facts, specialty_slug=None, limit=match_limit)
        except Exception:
            matches = []
    matches = annotate_applicability(matches, _patient_dict(doc))
    na_from_matches = [
        m for m in matches if m.get("applicability") == "not_applicable"
    ]
    if na_from_matches:
        seen_na = {
            str(x.get("source_path") or x.get("protocol_id") or "")
            for x in not_applicable
        }
        for m in na_from_matches:
            key = str(m.get("source_path") or m.get("protocol_id") or "")
            if key and key not in seen_na:
                not_applicable.append(m)
                seen_na.add(key)
    matches = [m for m in matches if m.get("applicability") != "not_applicable"]

    icd_codes = facts.get("consultation", {}).get("icd10") or []
    diagnosis_texts = [d.raw_text for d in doc.diagnoses if d.raw_text]

    summary_meta: dict[str, Any] = {"analysis_mode": analysis_mode or "legacy"}
    effective_mode = analysis_mode or "legacy"
    summary_condition_ids: list[str] = []
    merged_rules: list[dict[str, Any]] = []
    merge_meta: dict[str, Any] = {}
    summaries: list[Any] = []
    plan = None

    try:
        from .protocol_summary import config as ps_cfg
        from .protocol_summary.method_selector import merge_rules_for_plan, resolve_analysis_plan
        from .protocol_summary.summary_resolver import discover_protocol_summaries
        from .protocol_summary.summary_to_rules import (
            protocol_rule_to_legacy_dict,
            summary_to_protocol_rules,
        )

        cfg = ps_cfg.protocol_summary_config
        if analysis_mode == "legacy":
            enabled = False
        elif analysis_mode in ("summary", "hybrid"):
            enabled = True
        else:
            enabled = cfg.enabled

        summary_active = enabled or analysis_mode in ("summary", "hybrid")
        if not summary_active:
            summary_meta.update({
                "analysis_mode": "legacy",
                "protocol_summary_used": False,
                "fallback_to_legacy": False,
                "legacy_result_available": True,
                "summary_result_available": False,
                "summary_diagnostics": [],
                "summary_protocol_ids": [],
            })
        else:
            matched_pids = [
                str(m.get("protocol_id") or m.get("card_id") or "")
                for m in matches
                if m.get("protocol_id") or m.get("card_id")
            ]

            discovered, diagnostics, summary_condition_ids = discover_protocol_summaries(
                icd_codes=icd_codes,
                diagnosis_texts=diagnosis_texts,
                matched_protocols=matches,
                specialty_slug=effective_slug,
            )

            plan = resolve_analysis_plan(
                mode=analysis_mode,
                matched_protocol_ids=matched_pids,
                discovered_summaries=discovered,
                summary_diagnostics=diagnostics,
                enabled=enabled,
            )
            effective_mode = plan.mode
            summaries = list(plan.usable_summaries)

            summary_rules_dicts: list[dict[str, Any]] = []
            if plan.use_summary and summaries:
                for s in summaries:
                    for pr in summary_to_protocol_rules(s):
                        summary_rules_dicts.append(protocol_rule_to_legacy_dict(pr))
                        if pr.condition_id:
                            summary_condition_ids.append(pr.condition_id)
                summary_condition_ids = list(dict.fromkeys(summary_condition_ids))

            legacy_catalog = collect_catalog_rules(
                facts,
                matched_protocols=matches,
                condition_ids=summary_condition_ids or None,
            )

            merged_rules, merge_meta = merge_rules_for_plan(
                plan, legacy_catalog, summary_rules_dicts,
            )

            summary_meta = {
                "analysis_mode": plan.mode,
                "protocol_summary_used": bool(plan.use_summary and summary_rules_dicts),
                "protocol_summary_status": summaries[0].review_status if summaries else None,
                "fallback_to_legacy": (
                    (plan.primary_source == "legacy" and plan.mode != "legacy")
                    or (plan.use_summary and not summary_rules_dicts and plan.fallback_to_legacy)
                ),
                "legacy_result_available": plan.use_legacy or plan.mode == "legacy",
                "summary_result_available": bool(summary_rules_dicts),
                "summary_diagnostics": diagnostics,
                "summary_protocol_ids": [s.protocol_id for s in summaries],
                "rules_count_by_source": merge_meta.get("rules_count_by_source"),
                "rule_conflicts": merge_meta.get("rule_conflicts"),
            }
            if plan.notes:
                summary_meta["limitations"] = plan.notes
            if not summary_rules_dicts and plan.mode in ("summary", "hybrid"):
                summary_meta["limitations"] = list(dict.fromkeys(
                    list(summary_meta.get("limitations") or [])
                    + [n for n in plan.notes if "not found" in n or "fallback" in n or "legacy only" in n]
                ))
    except Exception as exc:
        merged_rules = []
        summary_meta["summary_resolution_error"] = str(exc)[:200]

    legacy_comparison_comp: dict[str, Any] | None = None
    if plan and plan.compare_with_legacy and plan.use_summary:
        try:
            legacy_check = run_rule_checker(
                facts, matched_protocols=matches, include_catalog=True,
            )
            legacy_report = build_compliance_report(
                doc, matches=matches, rules_check=legacy_check,
                not_applicable_matches=not_applicable, analysis_mode="legacy",
            )
            legacy_comparison_comp = report_to_json(legacy_report, doc)
        except Exception:
            legacy_comparison_comp = None

    try:
        if effective_mode == "legacy" or not merged_rules:
            rules_check = run_rule_checker(facts, matched_protocols=matches)
        elif effective_mode == "summary" and plan and plan.use_summary:
            rules_check = run_rule_checker(
                facts,
                matched_protocols=matches,
                extra_rules=merged_rules,
                include_catalog=False,
                condition_ids=summary_condition_ids or None,
            )
        else:
            suppressed = frozenset(merge_meta.get("suppressed_legacy_rule_ids") or [])
            rules_check = run_rule_checker(
                facts,
                matched_protocols=matches,
                extra_rules=[r for r in merged_rules if r.get("rule_source") == "summary"],
                include_catalog=True,
                skip_rule_ids=suppressed,
                condition_ids=summary_condition_ids or None,
            )
        rules_check["analysis_mode"] = effective_mode
        rules_check["summary_rules_count"] = sum(
            1 for r in merged_rules if r.get("rule_source") == "summary"
        )
        rules_check["legacy_rules_count"] = sum(
            1 for r in merged_rules if r.get("rule_source") != "summary"
        )
    except Exception:
        rules_check = {}

    summary_meta["summary_source_refs"] = [
        r.model_dump(mode="json")
        for r in _summary_source_refs(summaries, merged_rules)
    ]
    summary_meta["legacy_source_refs"] = [
        r.model_dump(mode="json")
        for r in _legacy_source_refs(matches, rules_check.get("findings") or [])
    ]

    report = build_compliance_report(
        doc,
        matches=matches,
        rules_check=rules_check,
        not_applicable_matches=not_applicable,
        analysis_mode=effective_mode,
        summary_meta=summary_meta,
    )

    # manual_review issues из конфликтов summary vs legacy
    if merge_meta.get("rule_conflicts"):
        for c in merge_meta["rule_conflicts"]:
            if c.get("manual_review"):
                report.warnings.append(
                    ComplianceIssue(
                        issue_type="summary_legacy_conflict",
                        severity="high",
                        message_ru=(
                            f"Конфликт summary/legacy по правилу {c.get('rule_key')}: "
                            f"решение={c.get('resolution')}"
                        ),
                        field_target="protocol_compliance",
                    ),
                )

    if legacy_comparison_comp is not None:
        current_json = report_to_json(report, doc)
        summary_meta["method_comparison"] = _build_method_comparison(legacy_comparison_comp, current_json)
        report.method_comparison = summary_meta["method_comparison"]

    report.summary_source_refs = [SourceRef.model_validate(x) for x in summary_meta.get("summary_source_refs") or []]
    report.legacy_source_refs = [SourceRef.model_validate(x) for x in summary_meta.get("legacy_source_refs") or []]

    try:
        anchor = effective_slug or normalize_rubric_slug(doc.doctor_specialty)
        rubric_slugs: list[str] = []
        if anchor:
            rubric_slugs.append(anchor)
            if icd_rubric and icd_rubric not in rubric_slugs:
                rubric_slugs.append(icd_rubric)
        else:
            rubric_slugs = rubric_slugs_from_matches(matches)[:2]
        rubric_specifics = extract_rubric_specifics(doc.raw_text or raw_text, rubric_slugs)
    except Exception:
        rubric_specifics = {"rubrics": [], "by_rubric": {}, "measurements": {}}

    compliance_json = report_to_json(report, doc)
    out: dict[str, Any] = {
        "document": doc.model_dump(mode="json"),
        "matches": matches,
        "rules_check": rules_check,
        "compliance": compliance_json,
        "rubric_specifics": rubric_specifics,
    }
    if with_markdown:
        out["report_markdown"] = report_to_markdown(report, doc, rubric_specifics=rubric_specifics)
        out["report_html"] = report_to_html(report, doc, rubric_specifics=rubric_specifics)
    return out
