"""Medication extraction from KZ text."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.patient_medication_extraction import (
    extract_medications_from_text,
    medications_patient_summary,
)

FIXTURE = Path(__file__).parent / "fixtures" / "neurology_kz_adult.txt"


def test_neurology_medications_not_zero() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    meds = extract_medications_from_text(text)
    names = {m["name"].lower() for m in meds}
    assert "мидокалм" in names or any("мидокалм" in n for n in names)
    assert len(meds) >= 4
    summary = medications_patient_summary(meds)
    assert "0 назначений" not in summary.lower()
    assert "лечение назначено" in summary.lower()
