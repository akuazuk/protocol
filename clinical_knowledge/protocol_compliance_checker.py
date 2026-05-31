"""Проверка соответствия КЗ клиническим протоколам (ТЗ §12).

Оборачивает findings rule_checker и дополняет оценку лечения/диагноза
ссылками на протокол и фрагментами КЗ.
"""
from __future__ import annotations

import re
from typing import Any

from .consult_schema import (
    ComplianceIssue,
    ConsultationDocument,
    SourceRef,
    TreatmentAssessment,
)

_TREATMENT_RULE_TYPES = frozenset({"keyword_presence"})
_TREATMENT_KEYWORDS = re.compile(
    r"лечен|препарат|доз|мг|таб|капс|антикоаг|антибиот|ингибитор|терап",
    re.I,
)


def _source_ref(src: dict[str, Any] | None) -> SourceRef | None:
    if not src:
        return None
    path = src.get("source_path") or src.get("local_path")
    if not path:
        return None
    return SourceRef(
        local_path=str(path),
        protocol_id=str(src.get("protocol_id") or "") or None,
        section_title=src.get("section_title"),
        quote=(src.get("quote") or "")[:400] or None,
    )


def _is_treatment_finding(f: dict[str, Any]) -> bool:
    rt = str(f.get("rule_type") or "")
    if rt not in _TREATMENT_RULE_TYPES:
        return False
    blob = " ".join(
        str(x) for x in [f.get("message_ru"), f.get("keyword"), f.get("rule_id")] if x
    )
    return bool(_TREATMENT_KEYWORDS.search(blob))


def _snippet(doc: ConsultationDocument, *fields: str, limit: int = 200) -> list[str]:
    out: list[str] = []
    s = doc.sections
    mapping = {
        "diagnosis": s.diagnosis_text,
        "treatment": s.recommendations_treatment,
        "exams": s.recommendations_exams or s.exam_results,
        "complaints": s.complaints,
    }
    for f in fields:
        val = mapping.get(f)
        if val:
            out.append(val[:limit])
    if doc.raw_text and not out:
        out.append(doc.raw_text[:limit])
    return out


def findings_to_issues(
    rules_check: dict[str, Any] | None,
    doc: ConsultationDocument,
) -> list[ComplianceIssue]:
    """Преобразует failed findings в ComplianceIssue с source_refs."""
    issues: list[ComplianceIssue] = []
    for f in (rules_check or {}).get("findings") or []:
        if f.get("passed"):
            continue
        rt = str(f.get("rule_type") or "")
        sev = str(f.get("severity") or "warning")
        if sev not in ("critical", "high", "warning", "info", "low", "medium"):
            sev = "warning"
        msg = str(f.get("message_ru") or "Несоответствие правилу протокола.")
        cat = {
            "diagnosis_formula": "diagnosis_protocol",
            "diagnostic_criterion": "clinical_criteria",
            "required_exam": "required_exams",
            "keyword_presence": "treatment_protocol",
            "population_mismatch": "protocol_applicability",
        }.get(rt, "protocol_compliance")
        refs: list[SourceRef] = []
        ref = _source_ref(f.get("source") if isinstance(f.get("source"), dict) else None)
        if ref:
            refs.append(ref)
        evidence = _snippet(
            doc,
            "diagnosis" if rt in ("diagnosis_formula", "diagnostic_criterion") else "treatment",
            "exams" if rt == "required_exam" else "diagnosis",
        )
        issues.append(
            ComplianceIssue(
                issue_type=str(f.get("rule_id") or rt),
                severity=sev,  # type: ignore[arg-type]
                category=cat,
                message_ru=msg,
                field_target=cat,
                source_refs=refs,
                consultation_evidence=evidence,
            )
        )
    return issues


def enhance_treatment_assessments(
    doc: ConsultationDocument,
    rules_check: dict[str, Any] | None,
    base: list[TreatmentAssessment],
) -> tuple[list[TreatmentAssessment], float | None]:
    """Дополняет оценку лечения правилами протокола (keyword_presence и др.)."""
    findings = [
        f for f in ((rules_check or {}).get("findings") or [])
        if _is_treatment_finding(f)
    ]
    out = list(base)
    scores: list[float] = []

    for t in base:
        penalty = sum(
            25 if i.issue_type == "missing_dose" else
            20 if i.issue_type == "missing_frequency" else 15
            for i in t.issues
        )
        scores.append(max(0.0, 100.0 - penalty))

    for f in findings:
        ok = bool(f.get("passed"))
        scores.append(100.0 if ok else 35.0)
        if not ok:
            refs: list[SourceRef] = []
            ref = _source_ref(f.get("source") if isinstance(f.get("source"), dict) else None)
            if ref:
                refs.append(ref)
            kw = str(f.get("keyword") or f.get("message_ru") or "протокол")[:120]
            out.append(
                TreatmentAssessment(
                    treatment_text=kw,
                    status="not_in_protocol",  # type: ignore[arg-type]
                    issues=[
                        ComplianceIssue(
                            issue_type=str(f.get("rule_id") or "protocol_treatment"),
                            severity=str(f.get("severity") or "warning"),  # type: ignore[arg-type]
                            category="treatment_protocol",
                            message_ru=str(f.get("message_ru") or f"Ожидалось по протоколу: {kw}"),
                            field_target="treatment",
                            source_refs=refs,
                            consultation_evidence=_snippet(doc, "treatment"),
                        )
                    ],
                    protocol_evidence=[str(f.get("message_ru") or kw)],
                    consultation_evidence=_snippet(doc, "treatment"),
                    source_refs=refs,
                )
            )

    if not scores:
        return out, None
    return out, round(sum(scores) / len(scores), 1)


def run_protocol_compliance_check(
    doc: ConsultationDocument,
    rules_check: dict[str, Any] | None,
    treatment_base: list[TreatmentAssessment],
) -> tuple[list[ComplianceIssue], list[TreatmentAssessment], float | None]:
    """Единая точка §12: issues из правил + усиленная оценка лечения."""
    issues = findings_to_issues(rules_check, doc)
    treatments, score = enhance_treatment_assessments(doc, rules_check, treatment_base)
    return issues, treatments, score
