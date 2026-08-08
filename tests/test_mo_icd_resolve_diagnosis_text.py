"""Diagnosis text only from clinical_diagnosis / mis_diagnos slots."""
from __future__ import annotations

from clinical_knowledge.mo_icd_resolve import resolve_diagnosis_text_from_mo


def test_prefers_diagnosis_slots() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": "Острый цистит",
            "objective_status": "N30.0 что-то ещё в статусе",
        }
    )
    assert got["text"] == "Острый цистит"
    assert got["source"] == "slots"
    assert got["used_fallback"] is False


def test_uses_clinical_and_mis_slots() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": (
                "Внебольничная нижнедолевая полисегментарная пневмония, "
                "средней тяжести.ДН0"
            ),
            "mis_diagnos": "J18.9",
            "anamnesis_doctor": "Ранее гипертония I10",
        }
    )
    assert "пневмония" in got["text"].lower()
    assert "J18.9" in got["text"] or got.get("main") == "J18.9"
    assert got["source"] == "slots"
    assert got["used_fallback"] is False


def test_no_fallback_from_non_diag_fields() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": "",
            "mis_diagnos": "",
            "objective_status": "Жалоб нет.\nДиагноз: Хронический тонзиллит J35.0\nПлан: полоскание.",
            "treatment_recommendations": "Код N30.0 Острый цистит. Пить больше жидкости.",
        }
    )
    assert got["text"] == ""
    assert got["source"] == "empty"
    assert got["used_fallback"] is False


def test_near_code_only_inside_diag_slot() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": "N30.0 Острый цистит",
            "mkb_code_main": "N30.0",
        }
    )
    assert "цистит" in got["text"].lower()
    assert got["source"] == "slots"
