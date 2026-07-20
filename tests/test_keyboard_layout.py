"""Авто-исправление неправильной раскладки в клинических запросах."""
from __future__ import annotations

from clinical_knowledge.keyboard_layout import (
    fix_keyboard_layout,
    has_cyrillic,
    maybe_fix_layout,
)


def test_fixes_temperature_and_cough() -> None:
    out, changed = maybe_fix_layout("ntvgthfnehf b rfitkm")
    assert changed is True
    assert out == "температура и кашель"


def test_fixes_pain_in_stomach_lowercase() -> None:
    out, changed = maybe_fix_layout(",jkm d ;bdjnt")
    assert changed is True
    assert out == "боль в животе"


def test_fixes_cough_in_child() -> None:
    out, changed = maybe_fix_layout("rfitkm e ht,tyrf")
    assert changed is True
    assert out == "кашель у ребенка"


def test_keeps_latin_drug_name() -> None:
    out, changed = maybe_fix_layout("amoxicillin 500")
    assert changed is False
    assert out == "amoxicillin 500"


def test_keeps_icd_code() -> None:
    out, changed = maybe_fix_layout("J45")
    assert changed is False
    assert out == "J45"


def test_keeps_real_russian_query() -> None:
    out, changed = maybe_fix_layout("боль в горле")
    assert changed is False
    assert out == "боль в горле"


def test_does_not_touch_mixed_cyrillic() -> None:
    out, changed = maybe_fix_layout("боль amoxicillin")
    assert changed is False
    assert out == "боль amoxicillin"


def test_punctuation_maps_lowercase() -> None:
    # ; -> ж, , -> б; не должно давать заглавные из-за верхнего регистра пунктуации
    assert fix_keyboard_layout(";,") == "жб"


def test_has_cyrillic() -> None:
    assert has_cyrillic("boль") is True
    assert has_cyrillic("bol") is False
