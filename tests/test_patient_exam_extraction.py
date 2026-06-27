"""Exam extraction from KZ text."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.patient_exam_extraction import extract_exams_from_text, exams_patient_summary

FIXTURE = Path(__file__).parent / "fixtures" / "neurology_kz_adult.txt"


def test_mri_recognized() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    exams = extract_exams_from_text(text)
    assert exams
    assert any(e.get("exam_type") == "MRI" for e in exams)
    summary = exams_patient_summary(exams)
    assert "0 обследован" not in summary.lower()
    assert "мрт" in summary.lower()
