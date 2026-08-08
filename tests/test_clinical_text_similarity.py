"""Shared clinical text similarity (foundation for ICD name_only + future section align)."""
from __future__ import annotations

from clinical_knowledge.clinical_text_similarity import (
    combined_score,
    fuzz_ratio,
    normalize_for_match,
    score_against_sections,
    strip_icd_codes,
    strip_leading_code_from_title,
    token_jaccard,
)


def test_strip_icd_codes_and_leading_title() -> None:
    assert "цистит" in strip_icd_codes("N30.0 Острый цистит").lower()
    assert "N30" not in strip_icd_codes("N30.0 Острый цистит").upper()
    clean = strip_leading_code_from_title("A00.0 - Холера, вызванная вибрионом")
    assert clean.startswith("Холера")
    assert "A00" not in clean


def test_normalize_and_typo_fuzz() -> None:
    a = normalize_for_match("Острый цистит N30.0")
    b = normalize_for_match("острый цистит")
    assert "n30" not in a
    assert fuzz_ratio(a, b) >= 0.9
    # опечатка: циститт
    assert fuzz_ratio("острый цистит", "острый циститт") >= 0.85


def test_combined_score_separates_unrelated() -> None:
    good = combined_score("Острый цистит", "Острый цистит")
    bad = combined_score("Острый цистит", "Инфаркт миокарда")
    assert good["combined"] >= 0.9
    assert bad["combined"] < 0.25
    assert token_jaccard("хронический тонзиллит", "тонзиллит хронический") >= 0.5


def test_score_against_sections_picks_supporting_slot() -> None:
    profile = score_against_sections(
        "Острый цистит",
        {
            "complaints": "дизурия, учащённое мочеиспускание, боль над лоном",
            "treatment": "амитриптилин на ночь",
        },
    )
    # жалобы ближе по смыслу токенов, чем лечение антидепрессантом
    assert profile["by_section"]["complaints"]["combined"] >= 0.0
    assert "best_section" in profile
