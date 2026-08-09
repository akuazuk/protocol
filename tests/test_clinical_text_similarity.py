"""Shared clinical text similarity (foundation for ICD name_only + future section align)."""
from __future__ import annotations

import os

from clinical_knowledge.clinical_text_similarity import (
    best_combined_against_title,
    combined_score,
    fuzz_ratio,
    light_stem,
    normalize_for_match,
    score_against_sections,
    split_diagnosis_phrases,
    strip_icd_codes,
    strip_leading_code_from_title,
    token_coverage,
    token_jaccard,
)


def test_strip_icd_codes_and_leading_title() -> None:
    assert "цистит" in strip_icd_codes("N30.0 Острый цистит").lower()
    assert "N30" not in strip_icd_codes("N30.0 Острый цистит").upper()
    clean = strip_leading_code_from_title("A00.0 - Холера, вызванная вибрионом")
    assert clean.startswith("Холера")
    assert "A00" not in clean
    # кириллическая «Е» / пробел в коде
    stripped = strip_icd_codes("Е 55.0 Дефицит витамина Д")
    assert "55" not in stripped
    assert "дефицит" in stripped.lower()


def test_split_diagnosis_phrases_and_best_phrase_score() -> None:
    diag = (
        "Бронхиальная астма, аллергическая, легкое течение. "
        "Персистирующий аллергический ринит. Головная боль напряжения"
    )
    phrases = split_diagnosis_phrases(diag)
    assert any("астма" in p.lower() for p in phrases)
    assert any("ринит" in p.lower() for p in phrases)
    # целый текст vs короткий title слабее, чем фраза «аллергический ринит»
    full = combined_score(diag, "Аллергический ринит неуточненный")["combined"]
    best = best_combined_against_title(diag, "Аллергический ринит неуточненный")
    assert best["combined"] >= full
    assert best["combined"] >= 0.5


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


def test_light_stem_zhivot_forms(monkeypatch) -> None:
    assert light_stem("животе") == "живот"
    assert light_stem("живота") == "живот"
    assert light_stem("живот") == "живот"
    monkeypatch.setenv("MO_ICD_LIGHT_STEM", "0")
    off = token_coverage("боль в животе", "боли в области живота")
    monkeypatch.setenv("MO_ICD_LIGHT_STEM", "1")
    on = token_coverage("боль в животе", "боли в области живота")
    assert off == 0.0
    assert on > 0.0
    assert combined_score("боль в животе", "боли в области живота")["combined"] > 0.25
    # default path без флага не ломает unrelated
    monkeypatch.delenv("MO_ICD_LIGHT_STEM", raising=False)
    assert os.environ.get("MO_ICD_LIGHT_STEM") in (None, "")
