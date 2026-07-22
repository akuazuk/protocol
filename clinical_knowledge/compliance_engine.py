"""Детерминированный движок оценки соответствия КЗ протоколам (ТЗ разделы 13-19).

На входе: разобранное КЗ (ConsultationDocument), аннотированные матчи протоколов и
результат детерминированной проверки правил (rules_check из rule_checker).
На выходе: ComplianceReport с проверяемыми оценками и source_refs.

Принципы ТЗ: не занижать балл при нехватке данных; не считать неприменимые по
возрасту/полу правила; suspected-диагноз требует дообследования; критический red flag
без маршрутизации => manual_review_required.
"""
from __future__ import annotations

import os
import re

from typing import Any

from .consult_schema import (
    ComplianceIssue,
    ComplianceReport,
    ConsultationDocument,
    DiagnosisAssessment,
    ExamAssessment,
    ProtocolAssessment,
    ProtocolMatchResult,
    ScoreBreakdown,
    SectionQualityAssessment,
    SourceRef,
    StructuralAssessment,
    TreatmentAssessment,
)
from .confidence_scoring import apply_confidence_status, compute_confidence_score
from .evidence_map import build_evidence_map
from .medication_parser import looks_like_medication_item
from .protocol_compliance_checker import run_protocol_compliance_check
from .requirement_checker import run_requirement_check
from .safety_checker import apply_safety_cap_to_score, has_unhandled_critical, run_safety_checks
from .scoring import compute_overall, sync_score_aliases

try:
    import icd_mkb
except ImportError:
    icd_mkb = None  # type: ignore[assignment]

from .dispensary_regulations import follow_up_mentioned_in_text, lookup_follow_up_expectations

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
                diagnosis_id=m.get("diagnosis_id"),
                document_title=m.get("title"),
                source_path=m.get("source_path"),
                rubric_name=m.get("specialty_slug"),
                matched_condition=m.get("matched_condition") or m.get("title"),
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
            mkb_valid = False
            mkb_title = None
            if d.icd10_code and icd_mkb is not None:
                mkb_valid = icd_mkb.is_code_in_ru_reference(d.icd10_code)
                mkb_title = icd_mkb.ru_title(d.icd10_code)
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
            elif icd_mkb is not None and not mkb_valid:
                missing.append(f"Код {d.icd10_code} отсутствует в справочнике МКБ-10.")
                issues.append(
                    ComplianceIssue(
                        issue_type="invalid_icd_code",
                        severity="high",
                        message_ru=f"Код МКБ-10 {d.icd10_code} не найден в русском справочнике.",
                        field_target="diagnosis",
                    )
                )
            elif mkb_title:
                found.append(f"Код {d.icd10_code} — {mkb_title[:80]}.")
            if has_proto:
                status = "supported"
                if not any("Код" in x for x in found):
                    found.append("Найден применимый протокол по диагнозу/МКБ.")
                for m in appl[:1]:
                    if m.get("source_path"):
                        refs.append(SourceRef(local_path=m.get("source_path"), protocol_id=str(m.get("protocol_id") or "") or None))
                if icd_mkb is not None and d.icd10_code:
                    if mkb_valid:
                        scores.append(92.0 if mkb_title else 80.0)
                    else:
                        scores.append(35.0)
                else:
                    scores.append(90.0 if d.icd10_code else 70.0)
            else:
                status = "insufficient_data"
                missing.append("Применимый протокол для диагноза не найден.")
                if icd_mkb is not None and d.icd10_code:
                    scores.append(75.0 if mkb_valid else 35.0)
                elif d.icd10_code:
                    scores.append(65.0)

        out.append(
            DiagnosisAssessment(
                diagnosis_id=d.diagnosis_id,
                diagnosis_text=d.diagnosis_name or d.raw_text,
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
    doc: ConsultationDocument,
) -> tuple[list[ExamAssessment], float | None]:
    findings = (rules_check or {}).get("findings") or []
    exam_findings = [f for f in findings if (f.get("rule_type") == "required_exam")]
    performed_names = {
        (e.exam_name or "").lower().strip()
        for e in (doc.performed_exams or [])
        if e.exam_name
    }
    if not exam_findings:
        return [], None
    out: list[ExamAssessment] = []
    passed = 0
    for f in exam_findings:
        exam_label = str(f.get("exam") or f.get("message_ru") or "обследование")[:200]
        ok = bool(f.get("passed"))
        if not ok and performed_names:
            low = exam_label.lower()
            if any(p in low or low in p for p in performed_names if len(p) > 3):
                ok = True
        passed += 1 if ok else 0
        src = f.get("source") or {}
        out.append(
            ExamAssessment(
                protocol_rule_id=f.get("rule_id"),
                exam_name=exam_label,
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
    out: list[TreatmentAssessment] = []
    drug_meds = [m for m in doc.medications if looks_like_medication_item(m)]
    if not drug_meds:
        return out, None
    scores: list[float] = []
    for m in drug_meds:
        issues: list[ComplianceIssue] = []
        penalty = 0
        raw_low = (m.raw_text or "").lower()
        informal_freq = bool(re.search(r"утром|вечером|на\s+ночь|\d+\s*р/с|\d+\s*раз", raw_low))
        pack_duration = bool(re.search(r"№\s*\d+", raw_low))
        if m.dose_value is None and not re.search(r"\d+(?:[.,]\d+)?\s*(?:в/м|в/мыш|в/в|мг|мл)", raw_low):
            issues.append(
                ComplianceIssue(
                    issue_type="missing_dose", severity="warning",
                    category="data_quality",
                    message_ru="Назначение без распознанной дозы.", field_target="treatment",
                    consultation_evidence=[m.raw_text[:200]] if m.raw_text else [],
                )
            )
            penalty += 25
        if not m.frequency and not informal_freq:
            issues.append(
                ComplianceIssue(
                    issue_type="missing_frequency", severity="warning",
                    category="data_quality",
                    message_ru="Назначение без распознанной кратности.", field_target="treatment",
                    consultation_evidence=[m.raw_text[:200]] if m.raw_text else [],
                )
            )
            penalty += 20
        if not m.duration and not m.schedule and not pack_duration:
            issues.append(
                ComplianceIssue(
                    issue_type="missing_duration", severity="warning",
                    category="data_quality",
                    message_ru="Назначение без распознанной длительности.", field_target="treatment",
                    consultation_evidence=[m.raw_text[:200]] if m.raw_text else [],
                )
            )
            penalty += 15
        item_score = max(0.0, 100.0 - penalty)
        scores.append(item_score)
        status = "partially_matches_protocol" if issues else "insufficient_data"
        out.append(
            TreatmentAssessment(
                medication_id=m.medication_id,
                treatment_text=m.raw_text,
                status=status,  # type: ignore[arg-type]
                issues=issues,
                consultation_evidence=[m.raw_text],
            )
        )
    treat_score = round(sum(scores) / len(scores), 1) if scores else None
    return out, treat_score


def _follow_up_planned(doc: ConsultationDocument) -> bool:
    if doc.follow_up or doc.sections.follow_up_text:
        return True
    blob = "\n".join(
        x for x in [
            doc.sections.general_recommendations or "",
            doc.sections.recommendations_exams or "",
            doc.sections.recommendations_treatment or "",
        ] if x
    ).lower()
    markers = (
        "контрольн", "повторн", "осмотр ", "через месяц", "через недел",
        "контроль узи", "дата повтор", "явк",
    )
    return any(m in blob for m in markers)


def _follow_up_score(doc: ConsultationDocument, structural: StructuralAssessment) -> float | None:
    icd_codes = [d.icd10_code for d in doc.diagnoses if d.icd10_code]
    reg = lookup_follow_up_expectations(icd_codes)
    blob = "\n".join(
        x for x in [
            doc.sections.follow_up_text or "",
            doc.sections.general_recommendations or "",
            doc.sections.recommendations_treatment or "",
        ] if x
    )
    if doc.follow_up:
        blob += "\n" + "\n".join(f.raw_text or "" for f in doc.follow_up)

    if _follow_up_planned(doc) or follow_up_mentioned_in_text(blob, min_months=reg.get("min_interval_months")):
        return 92.0 if reg.get("follow_up_hints") else 90.0
    if "follow_up_scheduled" in structural.missing_conditional:
        return 40.0
    if doc.sections.recommendations_treatment or doc.medications:
        if reg.get("follow_up_hints"):
            return 48.0
        return 55.0
    return None


def _protocol_assessment(
    matches: list[dict[str, Any]], rules_check: dict[str, Any],
) -> ProtocolAssessment:
    appl = [m for m in matches if m.get("applicability") in ("applicable", "possibly_applicable")]
    top = max(appl, key=lambda m: float(m.get("match_score") or 0), default=None) if appl else None
    rc = (rules_check or {}).get("rules_compliance_pct")
    pct = float(rc) if isinstance(rc, (int, float)) else None
    summary = ""
    if top:
        summary = f"Топ протокол: {top.get('title') or top.get('protocol_id') or ' - '}"
        if pct is not None:
            summary += f"; соответствие правилам: {pct:.0f}%"
    return ProtocolAssessment(
        matched_count=len(matches),
        applicable_count=len(appl),
        top_protocol_id=str(top.get("protocol_id") or "") if top else None,
        top_protocol_title=str(top.get("title") or "") if top else None,
        rules_compliance_pct=pct,
        summary_ru=summary,
    )


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


def _safety_score(safety, *, has_content: bool = True) -> float | None:
    if not safety:
        # «нет red flags» - это кредит только при наличии содержательного КЗ;
        # для пустого/нечитаемого документа это не оценка, а отсутствие данных.
        return 100.0 if has_content else None
    if any(s.severity == "critical" and s.status != "handled" for s in safety):
        return 0.0
    unhandled = [s for s in safety if s.status != "handled"]
    if not unhandled:
        return 80.0
    partial = [s for s in unhandled if s.status == "partially_handled"]
    if partial and len(partial) == len(unhandled):
        worst = max(
            (s.severity for s in partial),
            default="medium",
            key=lambda sev: ["low", "medium", "high", "critical"].index(sev),
        )
        return {"low": 75.0, "medium": 65.0, "high": 55.0, "critical": 45.0}.get(worst, 60.0)
    worst = max((s.severity for s in unhandled), default="medium",
                key=lambda sev: ["low", "medium", "high", "critical"].index(sev))
    return {"low": 70.0, "medium": 55.0, "high": 45.0, "critical": 0.0}.get(worst, 55.0)


def _protocol_match_score(matches: list[dict[str, Any]]) -> float | None:
    appl = [m for m in matches if m.get("applicability") in ("applicable", "possibly_applicable")]
    if not matches:
        return None
    if not appl:
        return 20.0
    best = max(appl, key=lambda m: float(m.get("match_score") or 0))
    return 90.0 if best.get("applicability") == "applicable" else 65.0


def _oncology_workup_suspicion(doc: ConsultationDocument) -> bool:
    blob = (doc.raw_text or "").lower()
    return any(
        m in blob
        for m in (
            "опухолевое образование",
            "картина опухол",
            "нельзя исключить инваз",
            "подозрени на зло",
            "подозрени на рак",
        )
    )


def _apply_oncology_priority_score_caps(
    doc: ConsultationDocument,
    *,
    pm_score: float | None,
    treat_score: float | None,
    overall: float | None,
) -> tuple[float | None, float | None, float | None]:
    """K30/диспепсия не должна маскировать подозрение на ЗНО (gastro_1)."""
    if not _oncology_workup_suspicion(doc):
        return pm_score, treat_score, overall
    if pm_score is not None:
        pm_score = min(pm_score, 55.0)
    if treat_score is not None:
        treat_score = min(treat_score, 72.0)
    if overall is not None:
        overall = min(overall, 75.0)
    return pm_score, treat_score, overall


def _apply_sparse_neurology_score_caps(
    doc: ConsultationDocument,
    structural: StructuralAssessment,
    *,
    diag_score: float | None,
    treat_score: float | None,
    safety_score: float | None,
) -> tuple[float | None, float | None, float | None]:
    from .requirement_checker import _sparse_primary_neurology

    if not _sparse_primary_neurology(doc):
        return diag_score, treat_score, safety_score
    s = doc.sections
    missing_core = not (s.complaints and s.anamnesis and s.objective_status)
    if missing_core and structural.structural_score is not None:
        structural.structural_score = min(structural.structural_score, 35.0)
    has_therapy = bool(
        doc.sections.recommendations_treatment
        or any(looks_like_medication_item(m) for m in doc.medications)
    )
    if missing_core and has_therapy:
        if treat_score is not None:
            treat_score = min(treat_score, 50.0)
        if safety_score is not None:
            safety_score = min(safety_score, 55.0)
    if missing_core and diag_score is not None:
        diag_score = min(diag_score, 45.0)
    return diag_score, treat_score, safety_score


def _apply_concurrent_nsaid_score_caps(
    safety: list,
    *,
    treat_score: float | None,
    safety_score: float | None,
) -> tuple[float | None, float | None]:
    has_dual = any(
        getattr(s, "issue_type", None) == "drug_safety"
        and "нпвп" in (getattr(s, "finding_text", None) or "").lower()
        and getattr(s, "status", None) != "handled"
        for s in safety
    )
    if not has_dual:
        return treat_score, safety_score
    if treat_score is not None:
        treat_score = min(treat_score, 25.0)
    if safety_score is not None:
        safety_score = min(safety_score, 15.0)
    return treat_score, safety_score


def _doc_raw_text_len(doc: ConsultationDocument) -> int:
    """Длина исходного текста КЗ (extraction_quality или fallback на raw_text)."""
    eq = getattr(doc, "extraction_quality", None)
    n = int(getattr(eq, "raw_text_length", 0) or 0)
    if n <= 0:
        n = len((doc.raw_text or "").strip())
    return n


def build_compliance_report(
    doc: ConsultationDocument,
    matches: list[dict[str, Any]] | None = None,
    rules_check: dict[str, Any] | None = None,
    *,
    not_applicable_matches: list[dict[str, Any]] | None = None,
    analysis_mode: str = "legacy",
    summary_meta: dict[str, Any] | None = None,
    alignment_block_scores: dict[str, float] | None = None,
) -> ComplianceReport:
    """Собрать ComplianceReport из разобранного КЗ, матчей и результата проверки правил.

    alignment_block_scores (Э3): переопределяет блоки diagnosis/exams/treatment/follow_up
    оценками из детерминированных alignment-карточек (структурные items + семантика),
    до применения всех caps и compute_overall - чтобы улучшение блоков дошло до overall,
    но safety/oncology/neurology gates по-прежнему действовали.
    """
    matches = matches or []
    rules_check = rules_check or {}

    has_content = bool(
        doc.diagnoses
        or doc.extraction_quality.parsed_sections_count > 0
        or len((doc.raw_text or "").strip()) >= 40
    )

    safety = run_safety_checks(doc)
    structural, req_issues = run_requirement_check(doc)
    diag_assess, diag_score = _diagnosis_assessments(doc, matches)
    exam_assess, exams_score = _exam_assessments(rules_check, doc)
    treat_base, treat_score_base = _treatment_assessments(doc)
    proto_issues, treat_assess, treat_score = run_protocol_compliance_check(
        doc, rules_check, treat_base,
    )
    if treat_score is None:
        treat_score = treat_score_base
    # Э3: переопределяем блоки оценками alignment-карточек ДО применения caps,
    # чтобы safety/oncology/neurology gates по-прежнему ограничивали overall.
    if alignment_block_scores:
        _abs = alignment_block_scores
        if isinstance(_abs.get("diagnosis"), (int, float)):
            diag_score = float(_abs["diagnosis"])
        if isinstance(_abs.get("exams"), (int, float)):
            exams_score = float(_abs["exams"])
        if isinstance(_abs.get("treatment"), (int, float)):
            treat_score = float(_abs["treatment"])
    section_q, doc_score = _section_quality(doc)
    safety_score = _safety_score(safety, has_content=has_content)
    diag_score, treat_score, safety_score = _apply_sparse_neurology_score_caps(
        doc, structural, diag_score=diag_score, treat_score=treat_score, safety_score=safety_score,
    )
    treat_score, safety_score = _apply_concurrent_nsaid_score_caps(
        safety, treat_score=treat_score, safety_score=safety_score,
    )
    pm_score = _protocol_match_score(matches)
    follow_score = _follow_up_score(doc, structural)
    if alignment_block_scores and isinstance(alignment_block_scores.get("follow_up"), (int, float)):
        follow_score = float(alignment_block_scores["follow_up"])
    proto_assess = _protocol_assessment(matches, rules_check)
    pm_score, treat_score, _ = _apply_oncology_priority_score_caps(
        doc, pm_score=pm_score, treat_score=treat_score, overall=None,
    )
    if not has_content:
        doc_score = None  # type: ignore[assignment]
        structural.structural_score = None
        structural.patient_data_score = None
        treat_score = None
        diag_score = None
        exams_score = None
        follow_score = None
        pm_score = None

    breakdown = ScoreBreakdown(
        documentation_score=structural.structural_score,
        structural_score=structural.structural_score,
        patient_data_score=structural.patient_data_score,
        protocol_applicability_score=pm_score,
        protocol_match_score=pm_score,
        diagnosis_score=diag_score,
        required_exams_score=exams_score,
        treatment_score=treat_score,
        safety_score=safety_score,
        follow_up_score=follow_score,
        documentation_quality_score=doc_score,
    )
    has_protocol = bool(_applicable_matches(matches))
    force_manual = has_unhandled_critical(safety)
    overall, status = compute_overall(
        breakdown, force_manual_review=force_manual, has_protocol_data=has_protocol,
    )
    sparse_min = max(0, int(os.environ.get("CONSULT_MIN_TEXT_LEN", "0")))
    text_len = _doc_raw_text_len(doc)
    if sparse_min and text_len > 0 and text_len < sparse_min:
        status = "insufficient_data"
    _, _, overall = _apply_oncology_priority_score_caps(
        doc, pm_score=None, treat_score=None, overall=overall,
    )
    breakdown = sync_score_aliases(breakdown)
    breakdown.overall_score = overall
    # Ось B (клиническая согласованность): взвешенное среднее блоков diagnosis/exams/
    # treatment/follow_up (для дашборда 3 осей; overall по-прежнему по всем весам).
    _conc_w = {"diagnosis_score": 0.20, "required_exams_score": 0.15,
               "treatment_score": 0.15, "follow_up_score": 0.05}
    _conc_num = 0.0
    _conc_den = 0.0
    for _k, _w in _conc_w.items():
        _v = getattr(breakdown, _k, None)
        if isinstance(_v, (int, float)):
            _conc_num += float(_v) * _w
            _conc_den += _w
    if _conc_den > 0:
        breakdown.clinical_concordance_score = round(_conc_num / _conc_den, 1)

    evidence_map = build_evidence_map(
        doc, rules_check,
        patient={
            "age_years": doc.patient.age_years,
            "sex": doc.patient.sex,
            "pregnancy": doc.patient.pregnancy,
            "adult_or_child": doc.patient.adult_or_child,
        },
    )

    draft = ComplianceReport(
        consultation_id=doc.consultation_id,
        source_file=doc.source_file,
        overall_score=overall,
        overall_status=status,  # type: ignore[arg-type]
        score_breakdown=breakdown,
        protocol_matches=_protocol_matches(matches),
        not_applicable_protocols=_protocol_matches(not_applicable_matches or []),
        evidence_map=evidence_map,
    )
    confidence = compute_confidence_score(doc, draft, rules_check=rules_check)
    breakdown.confidence_score = confidence
    draft.confidence_score = confidence
    draft.score_breakdown = breakdown
    status = apply_confidence_status(status, confidence)
    draft.overall_status = status  # type: ignore[assignment]

    capped_score, cap_info = apply_safety_cap_to_score(draft.overall_score, safety)
    if cap_info.applied and capped_score is not None:
        draft.overall_score = capped_score
        draft.score_breakdown.overall_score = capped_score
    draft.safety_cap = cap_info

    limitations: list[str] = []
    if not has_protocol:
        limitations.append("Не найден применимый протокол - клиническая оценка ограничена.")
    if confidence < 55:
        limitations.append("Низкая уверенность разбора - рекомендуется ручная проверка.")
    draft.limitations = limitations

    missing_items: list[ComplianceIssue] = []
    major_items: list[ComplianceIssue] = []
    warnings: list[ComplianceIssue] = []
    critical: list[ComplianceIssue] = []
    for iss in req_issues:
        if iss.issue_type in structural.missing_required:
            missing_items.append(iss)
        elif iss.severity == "critical":
            critical.append(iss)
        elif iss.severity == "high":
            major_items.append(iss)
        else:
            warnings.append(iss)
    for iss in proto_issues:
        if iss.severity == "critical":
            critical.append(iss)
        elif iss.severity == "high":
            major_items.append(iss)
        else:
            warnings.append(iss)
    for a in diag_assess:
        for iss in a.issues:
            if iss.severity == "critical":
                critical.append(iss)
            elif iss.severity == "high":
                major_items.append(iss)
            else:
                warnings.append(iss)
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
        if s.status == "handled":
            continue
        iss = ComplianceIssue(
            issue_type=s.issue_type, severity=s.severity, message_ru=s.finding_text,
            field_target="safety",
        )
        if s.severity == "critical" and s.status == "not_handled":
            critical.append(iss)
        elif s.severity in ("critical", "high"):
            major_items.append(iss)
        else:
            warnings.append(iss)

    refs: list[SourceRef] = []
    for m in matches[:5]:
        if m.get("source_path"):
            refs.append(SourceRef(local_path=m.get("source_path"), protocol_id=str(m.get("protocol_id") or "") or None))

    draft.diagnosis_assessments = diag_assess
    draft.section_quality = section_q
    draft.structural_assessment = structural
    draft.protocol_assessment = proto_assess
    draft.exam_assessments = exam_assess
    draft.treatment_assessments = treat_assess
    draft.safety_assessments = safety
    draft.missing_required_items = missing_items
    draft.major_issues = major_items
    draft.warnings = warnings
    draft.critical_issues = critical
    draft.explanation = (
        f"Детерминированная оценка (score_source=deterministic); "
        f"confidence={confidence}%; статус: {draft.overall_status}."
    )
    draft.source_refs = refs
    sm = summary_meta or {}
    mode = sm.get("analysis_mode") or analysis_mode or "legacy"
    draft.analysis_mode = mode  # type: ignore[assignment]
    draft.protocol_summary_used = bool(sm.get("protocol_summary_used"))
    draft.protocol_summary_status = sm.get("protocol_summary_status")
    draft.fallback_to_legacy = bool(sm.get("fallback_to_legacy"))
    draft.legacy_result_available = sm.get("legacy_result_available", True)
    draft.summary_result_available = bool(sm.get("summary_result_available"))
    draft.method_comparison = sm.get("method_comparison")
    if sm.get("summary_source_refs"):
        draft.summary_source_refs = [
            SourceRef.model_validate(x) if isinstance(x, dict) else x
            for x in sm["summary_source_refs"]
        ]
    if sm.get("legacy_source_refs"):
        draft.legacy_source_refs = [
            SourceRef.model_validate(x) if isinstance(x, dict) else x
            for x in sm["legacy_source_refs"]
        ]
    draft.summary_diagnostics = list(sm.get("summary_diagnostics") or [])
    draft.rules_count_by_source = sm.get("rules_count_by_source")
    if sm.get("limitations"):
        draft.limitations = list(dict.fromkeys(list(draft.limitations) + list(sm["limitations"])))
    if draft.protocol_summary_used and draft.fallback_to_legacy:
        draft.limitations.append("Часть правил взята из legacy fallback - summary-карточка неполная.")
    return draft
