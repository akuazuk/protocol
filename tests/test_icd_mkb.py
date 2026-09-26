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


def test_rectal_bleeding_suggests_gi_not_foreign_body() -> None:
    q = "кровь в кале и шишки воспаленные в заднем проходе"
    codes = [s["code"] for s in suggest_icd_from_russian(q, max_results=8)]
    assert codes[0] in ("K64.9", "K62.5", "K92.2", "K92.1", "K62.9")
    assert "T18.5" not in codes[:4]
    assert "Y44.6" not in codes[:4]
    assert "X18" not in codes[:4]
    assert "Z91.7" not in codes[:4]
    assert any(c.startswith(("K64", "K62", "K92")) for c in codes[:4])


def test_short_word_kale_no_substring_false_positives() -> None:
    """«кале» не должно матчить «раскаленными» / «калечащие»."""
    from icd_mkb import ru_lexicon_scored_entries

    q = "кровь в кале"
    codes = [r["code"] for r in ru_lexicon_scored_entries(q)[:10]]
    assert "X18" not in codes
    assert "Z91.7" not in codes


def test_retired_i84_maps_to_k64() -> None:
    from icd_mkb import canonical_ru_code, is_code_in_ru_reference, ru_title

    assert canonical_ru_code("I84.9") == "K64.9"
    assert is_code_in_ru_reference("I84.9") is True
    title = (ru_title("I84.9") or "").lower()
    assert "геморрой" in title


def test_lexicon_prefilter_matches_full_scan() -> None:
    """Предфильтр `_row_may_score` не меняет результат: сравнение с полным перебором строк."""
    import icd_mkb

    def brute(text: str) -> list[tuple[str, float]]:
        words, qlow = icd_mkb._ru_lexicon_cache_key(text)
        best: dict[str, float] = {}
        for code, title, _tlow in icd_mkb._ru_terminal_title_rows():
            sc = icd_mkb._lexicon_score_one_row(list(words), qlow, code, title)
            if sc <= 0:
                continue
            n = icd_mkb._norm_icd_code(code)
            best[n] = max(best.get(n, 0.0), sc)
        return sorted(((c, round(s, 2)) for c, s in best.items()), key=lambda x: (-x[1], x[0]))

    for text in (
        "Артериальная гипертензия 2 степени, риск 3. Головные боли",
        "кровь в кале",
        "ОРВИ. Кашель, насморк, температура 38",
        "Мигрень",
        "Хромота у ребёнка, боль в бедре",
    ):
        fast = sorted(
            ((r["code"], float(r["lex_score"])) for r in icd_mkb.ru_lexicon_scored_entries(text)),
            key=lambda x: (-x[1], x[0]),
        )
        assert fast == brute(text), text


def test_ru_title_index_matches_linear_scan_semantics() -> None:
    """Индекс код -> название даёт то же, что первый проход по справочнику."""
    import icd_mkb

    def linear(code: str) -> str | None:
        c = icd_mkb.canonical_ru_code(code)
        for row in icd_mkb._ru_rows():
            if icd_mkb._norm_icd_code(row.get("code") or "") == c:
                return (row.get("title_ru") or "").strip() or None
        return None

    for code in ("I10", "i10", "K29.7", "I84.9", "J06.9", "N17-N19", "A00.-", "ZZZ", ""):
        assert icd_mkb.ru_title(code) == linear(code), code
    assert icd_mkb.ru_title("I10")


def test_candidate_rows_equal_prefilter_over_all_rows() -> None:
    import icd_mkb

    rows = icd_mkb._ru_terminal_title_rows()
    for text in ("гипертония", "кровь в кале", "Мигрень", "Открытая рана уха", "боль в горле"):
        words, qlow = icd_mkb._ru_lexicon_cache_key(text)
        expected = [i for i, (_c, _t, tlow) in enumerate(rows) if icd_mkb._row_may_score(list(words), qlow, tlow)]
        assert icd_mkb._candidate_row_indices(list(words), qlow) == expected, text
