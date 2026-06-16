"""Тесты лексикона МКБ и клинических подсказок по жалобам."""
from __future__ import annotations

from icd_mkb import (
    analyze_query_for_icd,
    suggest_icd_from_russian,
    strip_funnel_context_lines,
)


def test_cough_fever_suggests_respiratory_not_exotic_fever() -> None:
    q = "Температура 39 и сухой кашель"
    codes = [s["code"] for s in suggest_icd_from_russian(q, max_results=8)]
    assert codes[0] in ("J06.9", "J20.9", "R50.9", "R05")
    assert "A25" not in codes[:4]
    assert any(c.startswith(("J06", "J20", "R50", "R05")) for c in codes[:4])


def test_refined_lixhoradka_query_still_ok_with_hints() -> None:
    """Даже если LLM подменил «температура» на «лихорадка», штраф A** и hints держат ОРВИ."""
    q = "лихорадка 39 сухой кашель"
    codes = [s["code"] for s in suggest_icd_from_russian(q, max_results=8)]
    assert codes[0] in ("J06.9", "J20.9", "R50.9", "R05")
    assert "A25" not in codes[:4]


def test_analyze_uses_lexicon_query_not_refined_rag() -> None:
    full = "Температура 39 и сухой кашель\nКонтекст подбора: взрослое население"
    refined = "лихорадка 39 сухой кашель\nКонтекст подбора: взрослое население"
    original = strip_funnel_context_lines(
        "Температура 39 и сухой кашель"
    )
    analysis = analyze_query_for_icd(full, refined, lexicon_query=original)
    codes = [s["code"] for s in analysis.get("suggested") or []]
    assert codes
    assert codes[0] in ("J06.9", "J20.9", "R50.9", "R05")
    assert "A25" not in codes[:4]
