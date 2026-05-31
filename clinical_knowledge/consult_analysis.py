"""Высокоуровневый оркестратор структурного анализа КЗ.

Связывает: парсинг -> подбор протоколов с применимостью -> детерминированные правила
-> compliance engine -> отчёты. Единая точка входа для интеграции в пайплайн/CLI.
"""
from __future__ import annotations

from typing import Any

from .compliance_engine import build_compliance_report
from .condition_registry import infer_conditions_hints
from .consult_parser import parse_consultation
from .consult_report import report_to_json, report_to_markdown
from .consult_schema import ConsultationDocument
from .protocol_match import annotate_applicability, match_protocol_cards
from .rubric_extractors import extract_rubric_specifics, normalize_rubric_slug, rubric_slugs_from_matches
from .rule_checker import run_rule_checker


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
) -> dict[str, Any]:
    """Полный структурный разбор + оценка соответствия для одного КЗ.

    Возвращает dict с ключами: document, matches, rules_check, compliance (JSON-отчёт),
    report_markdown (опц.).
    """
    doc = parse_consultation(
        raw_text,
        consultation_id=consultation_id,
        source_file=source_file,
        source_file_type=source_file_type,
        demographics_meta=demographics_meta,
    )
    facts = facts_from_document(doc)

    try:
        matches = match_protocol_cards(facts, specialty_slug=specialty_slug, limit=match_limit)
    except Exception:
        matches = []
    matches = annotate_applicability(matches, _patient_dict(doc))

    try:
        rules_check = run_rule_checker(facts, matched_protocols=matches)
    except Exception:
        rules_check = {}

    report = build_compliance_report(doc, matches=matches, rules_check=rules_check)

    try:
        rubric_slugs = rubric_slugs_from_matches(matches)
        spec_slug = normalize_rubric_slug(specialty_slug) or normalize_rubric_slug(doc.doctor_specialty)
        if spec_slug and spec_slug not in rubric_slugs:
            rubric_slugs = [spec_slug, *rubric_slugs]
        rubric_specifics = extract_rubric_specifics(doc.raw_text or raw_text, rubric_slugs)
    except Exception:
        rubric_specifics = {"rubrics": [], "by_rubric": {}, "measurements": {}}

    out: dict[str, Any] = {
        "document": doc.model_dump(mode="json"),
        "matches": matches,
        "rules_check": rules_check,
        "compliance": report_to_json(report, doc),
        "rubric_specifics": rubric_specifics,
    }
    if with_markdown:
        out["report_markdown"] = report_to_markdown(report, doc, rubric_specifics=rubric_specifics)
    return out
