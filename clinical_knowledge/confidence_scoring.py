"""Расчёт confidence_score для оценки КЗ (ТЗ improve_kz §4, §8)."""
from __future__ import annotations

from typing import Any

from .consult_schema import ComplianceReport, ConsultationDocument


def compute_confidence_score(
    doc: ConsultationDocument,
    report: ComplianceReport,
    *,
    rules_check: dict[str, Any] | None = None,
) -> float:
    """0-100: насколько системе хватает данных для надёжной оценки."""
    factors: list[float] = []

    eq = doc.extraction_quality
    factors.append(min(100.0, eq.confidence * 100 + 20 * eq.parsed_sections_count))

    if doc.patient.age_years is not None or doc.patient.birth_date:
        factors.append(90.0)
    else:
        factors.append(40.0)

    if doc.diagnoses or doc.sections.diagnosis_text:
        factors.append(85.0)
    else:
        factors.append(20.0)

    if report.protocol_matches:
        appl = [m for m in report.protocol_matches if m.applicability in ("applicable", "possibly_applicable")]
        factors.append(90.0 if appl else 50.0)
    else:
        factors.append(25.0)

    rc = (rules_check or {}).get("findings") or []
    if rc:
        factors.append(80.0)
    else:
        factors.append(45.0)

    if eq.has_undefined:
        factors.append(30.0)

    return round(sum(factors) / len(factors), 1) if factors else 0.0


def apply_confidence_status(
    overall_status: str,
    confidence: float,
    *,
    low_threshold: float = 55.0,
) -> str:
    """Если confidence низкий - low_confidence (кроме manual_review / insufficient)."""
    if overall_status in ("manual_review_required", "insufficient_data", "insufficient_protocol_data"):
        return overall_status
    if confidence < low_threshold:
        return "low_confidence"
    return overall_status
