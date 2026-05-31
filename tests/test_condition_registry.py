"""Tests for shared condition registry (all specialties)."""
from __future__ import annotations

from clinical_knowledge.condition_registry import infer_conditions_hints, score_card_for_hint


def test_infer_bronchitis_from_text_and_icd():
    hints = infer_conditions_hints("острый бронхит кашель", ["J20.9"])
    assert "acute_bronchitis" in hints


def test_infer_diabetes_from_icd():
    hints = infer_conditions_hints("консультация", ["E11.9"])
    assert "diabetes_mellitus" in hints


def test_score_card_for_pneumonia_hint():
    s = score_card_for_hint(
        "pneumonia",
        "клинический протокол лечения пневмонии у взрослых",
        ["J18.9"],
    )
    assert s >= 28


def test_infer_sle_marker():
    hints = infer_conditions_hints("системная красная волчанка скв", [])
    assert "sle" in hints
