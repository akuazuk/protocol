"""Taxonomy: оцениваем только clinical_visit."""
from __future__ import annotations

from clinical_knowledge.mo_daily import classify_document_kind, is_scored_document_kind


def test_full_visit_is_clinical_visit() -> None:
    kind, reason = classify_document_kind(
        {
            "kz_kind": "kz",
            "doctor_specialization": "Терапевт",
            "service_names": "Консультация врача-терапевта",
            "complaints": "боль в горле",
            "objective_status": "зев гиперемирован",
            "clinical_diagnosis": "J06.9 ОРВИ",
            "treatment_recommendations": "полоскание",
        }
    )
    assert kind == "clinical_visit"
    assert is_scored_document_kind(kind)
    assert "клинический" in reason.lower()


def test_stomatology_out_of_score() -> None:
    kind, _ = classify_document_kind(
        {
            "kz_kind": "kz",
            "doctor_specialization": "стоматолог-терапевт",
            "complaints": "боль",
            "clinical_diagnosis": "K02",
            "objective_status": "кариес",
        }
    )
    assert kind == "non_clinical"
    assert not is_scored_document_kind(kind)


def test_procedure_session_short_diagnosis() -> None:
    kind, reason = classify_document_kind(
        {
            "kz_kind": "kz",
            "doctor_specialization": "ЛОР-врач",
            "service_names": "Промывание лакун миндалин",
            "complaints": "на промывание миндалин",
            "clinical_diagnosis": "Хронический тонзиллит",
            "manipulations": "Промыты миндалины р-ром фурацилина",
        }
    )
    assert kind == "procedure_session"
    assert not is_scored_document_kind(kind)
    assert "манипуляц" in reason.lower() or "процедур" in reason.lower()


def test_medical_exam_pay_type_not_scored() -> None:
    kind, _ = classify_document_kind(
        {
            "pay_type": "12",
            "doctor_specialization": "Офтальмолог",
            "clinical_diagnosis": "Z00.0",
            "objective_status": "осмотр",
        }
    )
    assert kind == "medical_exam"
    assert not is_scored_document_kind(kind)


def test_consult_plus_uzi_stays_clinical_visit() -> None:
    kind, _ = classify_document_kind(
        {
            "kz_kind": "kz",
            "doctor_specialization": "Гинеколог",
            "service_names": "Консультация врача-гинеколога | УЗИ органов малого таза",
            "complaints": "боль",
            "objective_status": "осмотр",
            "clinical_diagnosis": "N94",
        }
    )
    assert kind == "clinical_visit"
