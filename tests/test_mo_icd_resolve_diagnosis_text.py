"""Diagnosis text from slots or full-document fallback near ICD code."""
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


def test_label_line_fallback_when_slots_empty() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": "",
            "objective_status": "Жалоб нет.\nДиагноз: Хронический тонзиллит J35.0\nПлан: полоскание.",
        }
    )
    assert "тонзиллит" in got["text"].lower()
    assert got["used_fallback"] is True
    assert got["source"].startswith("label_line:")


def test_near_code_fallback() -> None:
    got = resolve_diagnosis_text_from_mo(
        {
            "clinical_diagnosis": "",
            "mkb_code_main": "",
            "treatment_recommendations": "Код N30.0 Острый цистит. Пить больше жидкости.",
        }
    )
    assert "цистит" in got["text"].lower()
    assert got["used_fallback"] is True
    assert "near_code" in got["source"]
    assert "N30.0" in (got.get("codes") or []) or got.get("main") == "N30.0"
