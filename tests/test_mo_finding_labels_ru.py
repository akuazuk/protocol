"""Русские названия кодов замечаний МО."""
from __future__ import annotations

from clinical_knowledge.mo_finding_labels_ru import finding_label_ru


def test_known_codes_have_russian_titles() -> None:
    assert "обследован" in finding_label_ru("B_exams_gap").lower()
    assert "МКБ" in finding_label_ru("B_icd_invalid")
    assert "наблюден" in finding_label_ru("A_missing_follow_up").lower()
    assert "лечен" in finding_label_ru("A_missing_treatment").lower()
    assert finding_label_ru("B_exams_gap", "B_exams_gap").startswith("Не отражены")


def test_keeps_existing_russian_title() -> None:
    assert finding_label_ru("B_exams_gap", "Уже понятный текст") == "Уже понятный текст"
