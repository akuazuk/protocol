"""Детерминированный движок оценки соответствия КЗ протоколам (ТЗ разделы 13-19).

На входе: разобранное КЗ (ConsultationDocument), аннотированные матчи протоколов и
результат детерминированной проверки правил (rules_check из rule_checker).
На выходе: ComplianceReport с проверяемыми оценками и source_refs.

Принципы ТЗ: не занижать балл при нехватке данных; не считать неприменимые по
возрасту/полу правила; suspected-диагноз требует дообследования; критический red flag
без маршрутизации => manual_review_required.
"""
from __future__ import annotations

from typing import Any

from .consult_schema import (
    ComplianceIssue,
    ComplianceReport,
    ConsultationDocument,
    DiagnosisAssessment,
    ExamAssessment,
    ProtocolMatchResult,
    ScoreBreakdown,
    SectionQualityAssessment,
    SourceRef,
    TreatmentAssessment,
)
from .safety_checker import has_unhandled_critical, run_safety_checks
from .scoring import compute_overall

_ROUTING_MARKERS = (
    "консультац", "направлен", "госпитализац", "маршрут", "дообследован",
    "повторн", "контрол", "узи", "биопси", "онколог",
)


def _icd_root(code: str | None) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _applicable_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in matches if m.get("applicability") in ("applicable", "possibly_applicable", None)]


def _protocol_matches(matches: list[dict[str, Any]]) -> list[ProtocolMatchResult]:
    out: list[ProtocolMatchResult] = []
    for m in matches:
        out.append(
            ProtocolMatchResult(
                protocol_id=str(m.get("protocol_id") or ""),
                document_title=m.get("title"),
                source_path=m.get("source_path"),
                match_score=float(m.get("match_score") or 0.0),
                match_reasons=list(m.get("match_reasons") or []),
                mismatch_reasons=list(m.get("mismatch_reasons") or []),
                applicability=m.get("applicability") or "unknown",
                source_refs=[SourceRef(local_path=m.get("source_path"), protocol_id=str(m.get("protocol_id") or "") or None)]
                if m.get("source_path") else [],
            )
        )
    return out


def _routing_present(doc: ConsultationDocument) -> bool:
    s = doc.sections
    blob = "\n".join(
        x for x in [s.recommendations_exams, s.recommendations_treatment, s.general_recommendations, s.follow_up_text]
        if x
    ).lower()
    if any(d.date or d.raw_text for d in doc.follow_up):
        return True
    return any(mk in blob for mk in _ROUTING_MARKERS)


def _diagnosis_assessments(
    doc: ConsultationDocument, matches: list[dict[str, Any]],
) -> tuple[list[DiagnosisAssessment], float | None]:
    appl = _applicable_matches(matches)
    match_roots: set[str] = set()
    for m in appl:
        for code in (m.get("icd10_primary") or []):
            match_roots.add(_icd_root(str(code)))
    routing = _routing_present(doc)

    out: list[DiagnosisAssessment] = []
    scores: list[float] = []
    for d in doc.diagnoses:
        issues: list[ComplianceIssue] = []
        found: list[str] = []
        missing: list[str] = []
        refs: list[SourceRef] = []
        status = "not_assessed"

        if d.certainty == "suspected":
            status = "suspected_needs_confirmation"
            if routing:
                found.append("Назначено дообследование/маршрутизация для уточнения.")
                scores.append(70.0)
            else:
                missing.append("Дообследование для подтверждения подозрительного диагноза.")
                issues.append(
                    ComplianceIssue(
                        issue_type="suspected_without_workup",
                        severity="high",
                        message_ru="Подозрительный диагноз без назначенного дообследования.",
                        field_target="diagnosis",
                    )
                )
                scores.append(30.0)
        elif d.certainty == "excluded":
            status = "not_assessed"
        else:  # confirmed
            root = _icd_root(d.icd10_code)
            has_proto = bool(root and root in match_roots) or bool(appl)
            if not d.icd10_code:
                missing.append("Код МКБ-10 для диагноза.")
                issues.append(
                    ComplianceIssue(
                        issue_type="missing_icd",
                        severity="warning",
                        message_ru="У диагноза не указан код МКБ-10.",
                        field_target="diagnosis",
                    )
                )
            if has_proto:
                status = "supported"
                found.append("Найден применимый протокол по диагнозу/МКБ.")
                for m in appl[:1]:
                    if m.get("source_path"):
                        refs.append(SourceRef(local_path=m.get("source_path"), protocol_id=str(m.get("protocol_id") or "") or None))
                scores.append(90.0 if d.icd10_code else 70.0)
            else:
                status = "insufficient_data"
                missing.append("Применимый протокол для диагноза не найден.")

        out.append(
            DiagnosisAssessment(
                diagnosis_id=d.diagnosis_id,
                diagnosis_text=d.raw_text,
                icd10_code=d.icd10_code,
                status=status,  # type: ignore[arg-type]
                issues=issues,
                evidence_found=found,
                evidence_missing=missing,
                source_refs=refs,
            )
        )

    diag_score = round(sum(scores) / len(scores), 1) if scores else None
    return out, diag_score


def _exam_assessments(
    rules_check: dict[str, Any],
) -> tuple[list[ExamAssessment], float | None]:
    findings = (rules_check or {}).get("findings") or []
    exam_findings = [f for f in findings if (f.get("rule_type") == "required_exam")]
    if not exam_findings:
        return [], None
    out: list[ExamAssessment] = []
    passed = 0
    for f in exam_findings:
        ok = bool(f.get("passed"))
        passed += 1 if ok else 0
        src = f.get("source") or {}
        out.append(
            ExamAssessment(
                protocol_rule_id=f.get("rule_id"),
                exam_name=str(f.get("exam") or f.get("message_ru") or "обследование")[:200],
                status="present_performed" if ok else "missing_required",
                reason=str(f.get("message_ru") or ""),
                source_refs=[SourceRef(local_path=src.get("source_path"), protocol_id=src.get("protocol_id"))]
                if src.get("source_path") else [],
            )
        )
    score = round(passed / len(exam_findings) * 100, 1)
    return out, score


def _treatment_assessments(
    doc: ConsultationDocument,
) -> tuple[list[TreatmentAssessment], float | None]:
    # Без протокольных правил по препаратам не оцениваем дозы детерминированно:
    # фиксируем назначения как insufficient_data (не занижаем общий балл).
    out: list[TreatmentAssessment] = []
    for m in doc.medications:
        issues: list[ComplianceIssue] = []
        if m.dose_value is None:
            issues.append(
                ComplianceIssue(
                    issue_type="missing_dose", severity="warning",
                    message_ru="Назначение без распознанной дозы.", field_target="treatment",
                )
            )
        out.append(
            TreatmentAssessment(
                medication_id=m.medication_id,
                treatment_text=m.raw_text,
                status="insufficient_data",
                issues=issues,
                consultation_evidence=[m.raw_text],
            )
        )
    return out, None


def _section_quality(doc: ConsultationDocument) -> tuple[SectionQualityAssessment, float]:
    s = doc.sections
    checks = {
        "complaints": bool(s.complaints),
        "anamnesis": bool(s.anamnesis),
        "objective_status": bool(s.objective_status),
        "exam_results": bool(s.exam_results),
        "diagnosis": bool(s.diagnosis_text) or bool(doc.diagnoses),
        "recommendations": bool(s.recommendations_exams or s.general_recommendations),
        "treatment": bool(s.recommendations_treatment) or bool(doc.medications),
        "follow_up": bool(s.follow_up_text) or bool(doc.follow_up),
    }
    mandatory = ["complaints", "anamnesis", "objective_status", "diagnosis", "treatment"]
    missing = [k for k in checks if not checks[k]]
    placeholders = []
    if doc.extraction_quality.has_undefined:
        placeholders.append("undefined")

    present_mandatory = sum(1 for k in mandatory if checks[k])
    score = present_mandatory / len(mandatory) * 100
    if doc.extraction_quality.has_undefined:
        score = max(0.0, score - 20)

    sq = SectionQualityAssessment(
        has_complaints=checks["complaints"],
        has_anamnesis=checks["anamnesis"],
        has_objective_status=checks["objective_status"],
        has_exam_results=checks["exam_results"],
        has_diagnosis=checks["diagnosis"],
        has_recommendations=checks["recommendations"],
        has_treatment=checks["treatment"],
        has_follow_up=checks["follow_up"],
        missing_sections=missing,
        suspicious_placeholders=placeholders,
        extraction_warnings=list(doc.extraction_quality.warnings),
    )
    return sq, round(score, 1)


def _safety_score(safety) -> float | None:
    if not safety:
        return 100.0
    if any(s.severity == "critical" and s.status != "handled" for s in safety):
        return 0.0
    unhandled = [s for s in safety if s.status != "handled"]
    if not unhandled:
        return 80.0
    worst = max((s.severity for s in unhandled), default="medium",
                key=lambda sev: ["low", "medium", "high", "critical"].index(sev))
    return {"low": 70.0, "medium": 55.0, "high": 40.0, "critical": 0.0}.get(worst, 55.0)


def _protocol_match_score(matches: list[dict[str, Any]]) -> float | None:
    appl = [m for m in matches if m.get("applicability") in ("applicable", "possibly_applicable")]
    if not matches:
        return None
    if not appl:
        return 20.0
    best = max(appl, key=lambda m: float(m.get("match_score") or 0))
    return 90.0 if best.get("applicability") == "applicable" else 65.0


def build_compliance_report(
    doc: ConsultationDocument,
    matches: list[dict[str, Any]] | None = None,
    rules_check: dict[str, Any] | None = None,
) -> ComplianceReport:
    """Собрать ComplianceReport из разобранного КЗ, матчей и результата проверки правил."""
    matches = matches or []
    rules_check = rules_check or {}

    safety = run_safety_checks(doc)
    diag_assess, diag_score = _diagnosis_assessments(doc, matches)
    exam_assess, exams_score = _exam_assessments(rules_check)
    treat_assess, treat_score = _treatment_assessments(doc)
    section_q, doc_score = _section_quality(doc)
    safety_score = _safety_score(safety)
    pm_score = _protocol_match_score(matches)

    breakdown = ScoreBreakdown(
        protocol_match_score=pm_score,
        diagnosis_score=diag_score,
        required_exams_score=exams_score,
        treatment_score=treat_score,
        safety_score=safety_score,
        documentation_quality_score=doc_score,
    )
    force_manual = has_unhandled_critical(safety)
    overall, status = compute_overall(breakdown, force_manual_review=force_manual)
    breakdown.overall_score = overall

    missing_items: list[ComplianceIssue] = []
    warnings: list[ComplianceIssue] = []
    critical: list[ComplianceIssue] = []
    for a in diag_assess:
        for iss in a.issues:
            (critical if iss.severity in ("critical", "high") else warnings).append(iss)
    for ex in exam_assess:
        if ex.status == "missing_required":
            missing_items.append(
                ComplianceIssue(
                    issue_type="missing_required_exam", severity="warning",
                    message_ru=ex.reason or f"Отсутствует обязательное обследование: {ex.exam_name}",
                    field_target="exams", source_refs=ex.source_refs,
                )
            )
    for s in safety:
        if s.status != "handled":
            iss = ComplianceIssue(
                issue_type=s.issue_type, severity=s.severity, message_ru=s.finding_text,
                field_target="safety",
            )
            (critical if s.severity in ("critical", "high") else warnings).append(iss)

    refs: list[SourceRef] = []
    for m in matches[:5]:
        if m.get("source_path"):
            refs.append(SourceRef(local_path=m.get("source_path"), protocol_id=str(m.get("protocol_id") or "") or None))

    return ComplianceReport(
        consultation_id=doc.consultation_id,
        overall_score=overall,
        overall_status=status,  # type: ignore[arg-type]
        score_breakdown=breakdown,
        protocol_matches=_protocol_matches(matches),
        diagnosis_assessments=diag_assess,
        section_quality=section_q,
        exam_assessments=exam_assess,
        treatment_assessments=treat_assess,
        safety_assessments=safety,
        missing_required_items=missing_items,
        warnings=warnings,
        critical_issues=critical,
        explanation=(
            f"Детерминированная оценка по {sum(1 for v in [pm_score, diag_score, exams_score, treat_score, safety_score, doc_score] if v is not None)} "
            f"блокам; статус: {status}."
        ),
        source_refs=refs,
    )
