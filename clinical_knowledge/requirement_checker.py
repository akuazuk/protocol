"""Проверка обязательных и условных рубрик КЗ (ТЗ §9–10).

Отдельно от rule_checker (протокольные правила): здесь — структура и качество
заполнения документа по нормативным/внутренним требованиям.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .consult_schema import (
    ComplianceIssue,
    ConsultationDocument,
    StructuralAssessment,
)
from .diagnosis_icd import lookup_disease_icd, normalize_code
from .safety_checker import run_safety_checks

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

_DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "required": [
        {"id": "consultation_date", "field": "consultation_date", "title_ru": "Дата консультации", "severity": "major"},
        {"id": "doctor_specialty", "field": "doctor_specialty", "title_ru": "Специальность врача", "severity": "major"},
        {"id": "patient_identity", "field": "patient.full_name", "title_ru": "Данные пациента", "severity": "major"},
        {"id": "patient_age", "field": "patient.age_years", "alt_field": "patient.birth_date", "title_ru": "Возраст или ДР", "severity": "major"},
        {"id": "diagnosis", "field": "sections.diagnosis_text", "alt_field": "diagnoses", "title_ru": "Диагноз", "severity": "critical"},
        {"id": "objective_status", "field": "sections.objective_status", "title_ru": "Объективный статус", "severity": "major"},
        {"id": "recommendations", "field": "sections.recommendations_treatment", "alt_field": "sections.general_recommendations", "title_ru": "Рекомендации", "severity": "critical"},
        {"id": "doctor_identification", "field": "doctor_name", "alt_field": "sections.doctor_signature", "title_ru": "ФИО врача", "severity": "major"},
    ],
    "conditional": [],
    "recommended": [],
    "profile_specialties": ["дерматolog", "хирург", "флебolog", "офтальмolog", "лор"],
}


@lru_cache(maxsize=1)
def load_kz_requirements() -> dict[str, Any]:
    path = CONFIG_DIR / "kz_requirements.yaml"
    try:
        import yaml  # type: ignore
    except ImportError:
        return _DEFAULT_REQUIREMENTS
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else _DEFAULT_REQUIREMENTS
    except (OSError, ValueError):
        return _DEFAULT_REQUIREMENTS


def _resolve_field(doc: ConsultationDocument, path: str) -> Any:
    if not path:
        return None
    if path == "diagnoses":
        return doc.diagnoses or None
    if path == "diagnoses.icd10":
        codes = [d.icd10_code for d in doc.diagnoses if d.icd10_code]
        return codes or None
    if path == "follow_up":
        return doc.follow_up or None
    if path == "performed_exams":
        return doc.performed_exams or None
    if path == "clinic_name":
        return doc.clinic_name
    parts = path.split(".")
    obj: Any = doc
    for p in parts:
        if obj is None:
            return None
        if p == "sections":
            obj = doc.sections
        elif p == "patient":
            obj = doc.patient
        elif hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            return None
    if isinstance(obj, str):
        return obj.strip() or None
    if isinstance(obj, list):
        return obj if obj else None
    return obj


def _present(doc: ConsultationDocument, field: str, alt_field: str | None = None) -> bool:
    val = _resolve_field(doc, field)
    if val:
        return True
    if alt_field:
        return bool(_resolve_field(doc, alt_field))
    return False


def _is_profile_specialty(doc: ConsultationDocument, needles: list[str]) -> bool:
    blob = " ".join(
        x for x in [doc.doctor_specialty or "", doc.raw_text[:800] if doc.raw_text else ""] if x
    ).lower()
    return any(n.lower() in blob for n in needles)


def _primary_visit(doc: ConsultationDocument) -> bool:
    blob = (doc.raw_text or "").lower()
    if re.search(r"повторн\w*\s+(?:консультац|при[её]м|осмотр)|на\s+контрол", blob):
        return False
    return bool(doc.sections.complaints or doc.sections.anamnesis)


def _follow_up_mentioned(doc: ConsultationDocument) -> bool:
    blob = (doc.raw_text or "").lower()
    return bool(
        doc.follow_up
        or doc.sections.follow_up_text
        or re.search(r"контрол\w*\s+(?:через|через\s+\d)|повторн\w*\s+явк", blob)
    )


def _suspected_diagnosis(doc: ConsultationDocument) -> bool:
    return any(d.certainty == "suspected" for d in doc.diagnoses) or bool(
        doc.extraction_quality.has_question_mark_diagnosis
    )


def _exam_supported_diagnosis(doc: ConsultationDocument) -> bool:
    if doc.performed_exams or doc.sections.exam_results:
        return True
    blob = (doc.sections.diagnosis_text or "").lower()
    return bool(re.search(r"по\s+(?:данным|результатам)|подтвержд|фгдс|узи|кт|мрт|биопси", blob))


def _transfer_mentioned(doc: ConsultationDocument) -> bool:
    blob = (doc.raw_text or "").lower()
    return bool(re.search(r"перевод|транспорт|санавиац|эвакуац", blob))


def _routing_present(doc: ConsultationDocument) -> bool:
    if doc.sections.routing:
        return True
    blob = "\n".join(
        x for x in [
            doc.sections.recommendations_exams or "",
            doc.sections.recommendations_treatment or "",
            doc.sections.general_recommendations or "",
            doc.sections.follow_up_text or "",
        ] if x
    ).lower()
    return any(m in blob for m in ("направлен", "госпитализац", "маршрут", "дообследован", "консультац"))


def _issue(
    req_id: str,
    title: str,
    *,
    severity: str,
    category: str = "missing_required_section",
    evidence: list[str] | None = None,
) -> ComplianceIssue:
    sev_map = {"minor": "warning", "major": "high", "critical": "critical", "info": "info"}
    return ComplianceIssue(
        issue_type=req_id,
        severity=sev_map.get(severity, severity),  # type: ignore[arg-type]
        category=category,
        message_ru=f"Отсутствует или не распознано: {title}.",
        field_target=req_id,
        expected=title,
        consultation_evidence=evidence or [],
    )


def _data_quality_issues(doc: ConsultationDocument) -> list[ComplianceIssue]:
    issues: list[ComplianceIssue] = []
    raw_snip = [(doc.raw_text or "")[:240]] if doc.raw_text else []

    if doc.extraction_quality.has_undefined:
        issues.append(
            ComplianceIssue(
                issue_type="undefined_placeholder",
                severity="high",
                category="data_quality",
                message_ru="В тексте КЗ найдено «undefined» — дефект качества данных.",
                field_target="raw_text",
                actual="undefined",
                consultation_evidence=raw_snip,
            )
        )
    if doc.patient.sex == "unknown":
        issues.append(
            ComplianceIssue(
                issue_type="missing_sex",
                severity="warning",
                category="data_quality",
                message_ru="Пол пациента не указан или не распознан.",
                field_target="patient.sex",
                consultation_evidence=raw_snip,
            )
        )
    if not doc.doctor_specialty:
        issues.append(
            ComplianceIssue(
                issue_type="missing_doctor_specialty",
                severity="warning",
                category="data_quality",
                message_ru="Не распознана специальность врача.",
                field_target="doctor_specialty",
                consultation_evidence=raw_snip,
            )
        )
    if doc.extraction_quality.has_missing_birth_date and doc.patient.age_years is None:
        issues.append(
            ComplianceIssue(
                issue_type="missing_age",
                severity="warning",
                category="data_quality",
                message_ru="Не указаны дата рождения и возраст пациента.",
                field_target="patient",
                consultation_evidence=raw_snip,
            )
        )
    if doc.extraction_quality.has_missing_consultation_date:
        issues.append(
            ComplianceIssue(
                issue_type="missing_consultation_date",
                severity="high",
                category="data_quality",
                message_ru="Не распознана дата консультации.",
                field_target="consultation_date",
                consultation_evidence=raw_snip,
            )
        )
    for d in doc.diagnoses:
        if d.certainty == "confirmed" and not d.icd10_code and d.raw_text:
            issues.append(
                ComplianceIssue(
                    issue_type="diagnosis_without_icd",
                    severity="warning",
                    category="data_quality",
                    message_ru=f"Диагноз без кода МКБ-10: {d.raw_text[:100]}",
                    field_target="diagnosis",
                    actual=d.raw_text[:160],
                    consultation_evidence=[d.raw_text[:200]],
                )
            )
        if d.icd10_code and d.diagnosis_name:
            lex = lookup_disease_icd(d.diagnosis_name or d.raw_text)
            if lex:
                expected = normalize_code(lex[0])
                actual = normalize_code(d.icd10_code)
                if expected and actual and not actual.startswith(expected[:3]):
                    issues.append(
                        ComplianceIssue(
                            issue_type="icd_text_mismatch",
                            severity="warning",
                            category="data_quality",
                            message_ru=(
                                f"Код МКБ {actual} может не соответствовать тексту диагноза "
                                f"(ожидается ~{expected})."
                            ),
                            field_target="diagnosis",
                            expected=expected,
                            actual=actual,
                            consultation_evidence=[d.raw_text[:200]],
                        )
                    )
    exam_blob = doc.sections.exam_results or ""
    if exam_blob and not re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2}", exam_blob):
        if re.search(r"оак|узи|фгдс|кт|мрт|экг|анализ", exam_blob, re.I):
            issues.append(
                ComplianceIssue(
                    issue_type="exams_without_dates",
                    severity="warning",
                    category="data_quality",
                    message_ru="Обследования перечислены без дат выполнения.",
                    field_target="exam_results",
                    consultation_evidence=[exam_blob[:200]],
                )
            )
    if _follow_up_mentioned(doc) and not (doc.follow_up or doc.sections.follow_up_text):
        issues.append(
            ComplianceIssue(
                issue_type="follow_up_without_date",
                severity="warning",
                category="data_quality",
                message_ru="Упомянут контроль/повторная явка без конкретной даты или срока.",
                field_target="follow_up",
                consultation_evidence=raw_snip,
            )
        )
    for m in doc.medications:
        ev = [m.raw_text[:200]] if m.raw_text else []
        if m.dose_value is None and m.raw_text:
            issues.append(
                ComplianceIssue(
                    issue_type="missing_dose",
                    severity="warning",
                    category="data_quality",
                    message_ru=f"Назначение без распознанной дозы: {m.raw_text[:120]}",
                    field_target="treatment",
                    actual=m.raw_text[:160],
                    consultation_evidence=ev,
                )
            )
        if not m.frequency and m.raw_text and re.search(r"\d+\s*мг", m.raw_text, re.I):
            issues.append(
                ComplianceIssue(
                    issue_type="missing_frequency",
                    severity="warning",
                    category="data_quality",
                    message_ru=f"Назначение без распознанной кратности: {m.raw_text[:120]}",
                    field_target="treatment",
                    actual=m.raw_text[:160],
                    consultation_evidence=ev,
                )
            )
        if not m.duration and not m.schedule and m.raw_text:
            if re.search(r"(?:таб|капс|мг|мл)", m.raw_text, re.I):
                issues.append(
                    ComplianceIssue(
                        issue_type="missing_duration",
                        severity="warning",
                        category="data_quality",
                        message_ru=f"Назначение без распознанной длительности: {m.raw_text[:120]}",
                        field_target="treatment",
                        actual=m.raw_text[:160],
                        consultation_evidence=ev,
                    )
                )
    return issues


def run_requirement_check(doc: ConsultationDocument) -> tuple[StructuralAssessment, list[ComplianceIssue]]:
    """Проверка структуры КЗ: required / conditional / recommended + data quality."""
    cfg = load_kz_requirements()
    profile_needles = list(cfg.get("profile_specialties") or [])
    safety = run_safety_checks(doc)
    unhandled_critical = any(s.severity == "critical" and s.status != "handled" for s in safety)

    ctx = {
        "primary_visit": _primary_visit(doc),
        "profile_specialty": _is_profile_specialty(doc, profile_needles),
        "icd_expected": bool(doc.diagnoses) or bool(doc.sections.diagnosis_text),
        "follow_up_mentioned": _follow_up_mentioned(doc),
        "unhandled_red_flag": unhandled_critical,
        "suspected_diagnosis": _suspected_diagnosis(doc),
        "exam_supported_diagnosis": _exam_supported_diagnosis(doc),
        "transfer_mentioned": _transfer_mentioned(doc),
    }

    filled: list[str] = []
    missing_required: list[str] = []
    missing_conditional: list[str] = []
    missing_recommended: list[str] = []
    issues: list[ComplianceIssue] = []

    def process_tier(tier: str, items: list[dict[str, Any]], missing_list: list[str]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id") or "")
            when = item.get("when")
            if when and not ctx.get(str(when), False):
                continue
            field = str(item.get("field") or "")
            alt = item.get("alt_field")
            alt_s = str(alt) if alt else None
            title = str(item.get("title_ru") or rid)
            if _present(doc, field, alt_s):
                filled.append(rid)
            else:
                missing_list.append(rid)
                sev = str(item.get("severity") or ("major" if tier == "required" else "warning"))
                cat = "missing_required_section" if tier == "required" else "missing_conditional_section"
                issues.append(_issue(rid, title, severity=sev, category=cat))

    process_tier("required", list(cfg.get("required") or []), missing_required)
    process_tier("conditional", list(cfg.get("conditional") or []), missing_conditional)
    for item in list(cfg.get("recommended") or []):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or "")
        field = str(item.get("field") or "")
        if _present(doc, field):
            filled.append(rid)
        else:
            missing_recommended.append(rid)

    issues.extend(_data_quality_issues(doc))

    if ctx["unhandled_red_flag"] and not _routing_present(doc):
        snip = [(doc.raw_text or "")[:240]] if doc.raw_text else []
        issues.append(
            ComplianceIssue(
                issue_type="routing_red_flag",
                severity="critical",
                category="missing_conditional_section",
                message_ru="Критический красный флаг без маршрутизации/дообследования.",
                field_target="routing",
                consultation_evidence=snip,
            )
        )

    n_req = len(list(cfg.get("required") or [])) or 1
    structural_score = round(len([r for r in filled if r not in missing_recommended]) / max(n_req, 1) * 100, 1)
    structural_score = max(0.0, min(100.0, structural_score - 10 * len(missing_required)))

    patient_checks = [
        doc.patient.full_name,
        doc.patient.age_years or doc.patient.birth_date,
        doc.patient.sex if doc.patient.sex != "unknown" else None,
    ]
    patient_score = round(sum(1 for x in patient_checks if x) / 3 * 100, 1)

    assessment = StructuralAssessment(
        filled_sections=filled,
        missing_required=missing_required,
        missing_conditional=missing_conditional,
        missing_recommended=missing_recommended,
        structural_score=structural_score,
        patient_data_score=patient_score,
    )
    return assessment, issues
