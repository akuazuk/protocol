"""Тесты pydantic-моделей структурного анализа КЗ (этап 1)."""
from __future__ import annotations

from datetime import date

from clinical_knowledge.consult_schema import (
    ComplianceReport,
    ConsultationDiagnosis,
    ConsultationDocument,
    MedicationItem,
    MedicationScheduleStep,
    PatientContext,
)


def test_consultation_document_defaults():
    doc = ConsultationDocument(consultation_id="c1")
    assert doc.consultation_id == "c1"
    assert doc.patient.sex == "unknown"
    assert doc.diagnoses == []
    assert doc.extraction_quality.confidence == 0.0
    # сериализация не падает
    dumped = doc.model_dump(mode="json")
    assert dumped["consultation_id"] == "c1"


def test_extra_keys_ignored():
    # эвристики/LLM могут добавлять лишние ключи - это не должно ронять модель
    p = PatientContext.model_validate({"sex": "female", "unexpected": 123})
    assert p.sex == "female"
    assert not hasattr(p, "unexpected")


def test_diagnosis_and_medication():
    d = ConsultationDiagnosis(
        diagnosis_id="d1", raw_text="K30 Диспепсия", icd10_code="K30",
        certainty="suspected",
    )
    assert d.certainty == "suspected"
    m = MedicationItem(
        medication_id="m1", raw_text="Преднизолон 5 мг по 12 таб",
        drug_name="Преднизолон", dose_value=5.0, dose_unit="мг",
        schedule=[MedicationScheduleStep(dose_text="5 мг по 12 таб", start_date=date(2024, 8, 12))],
    )
    assert m.schedule[0].start_date == date(2024, 8, 12)


def test_compliance_report_default_status():
    rep = ComplianceReport(consultation_id="c1")
    assert rep.overall_status == "insufficient_data"
    assert rep.score_breakdown.overall_score is None
    rep.model_dump(mode="json")
