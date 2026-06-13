"""Тесты обезличивания ФИО."""
from __future__ import annotations

from clinical_knowledge.privacy import name_to_initials, redact_kz_text_for_display


def test_full_name_to_initials():
    assert name_to_initials("Кузавка Павел Леонидович") == "К. П. Л."
    assert name_to_initials("Иванов Павел Леонидович") == "И. П. Л."


def test_name_with_dob_suffix():
    assert name_to_initials("Кузавка Павел Леонидович, 12.07.1976") == "К. П. Л."


def test_empty_name():
    assert name_to_initials("") == " - "
    assert name_to_initials(None) == " - "


def test_redact_fio_line():
    text = "ФИО: Петров Петр Петрович, 01.01.1980\nЛечение: PPI"
    out = redact_kz_text_for_display(text)
    assert "Петров" not in out
    assert "PPI" in out or "Лечение" in out
