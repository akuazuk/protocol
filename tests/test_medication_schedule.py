"""Regression: prednisolone taper schedule (pl_2_d_s_2)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.medication_parser import parse_medications

FIX = Path(__file__).resolve().parent / "fixtures" / "consultations"


def test_prednisolone_schedule_prefix_lines():
    text = (FIX / "pl_2_d_s_2.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="pl_2_d_s_2")
    assert doc.medications
    med = doc.medications[0]
    assert "преднизолон" in (med.drug_name or med.raw_text or "").lower()
    assert len(med.schedule) >= 2


def test_prednisolone_schedule_suffix_semicolon():
    block = (
        "преднизолон 5 мг по 12 таблеток с 12.08.2024; "
        "4 мг по 12 таб с 19.08.2024; 3 мг по 12 таб с 26.08.2024"
    )
    meds = parse_medications(block)
    assert meds
    assert len(meds[0].schedule) >= 2
