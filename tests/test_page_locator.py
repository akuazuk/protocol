"""Локатор страницы PDF по дословной цитате (обогащение page_start в карточках)."""
from __future__ import annotations

from clinical_knowledge.page_locator import locate_page_for_quote

_CHUNKS = [
    {"text": "Титульный лист и оглавление протокола.", "page_from": 1},
    {
        "text": "Диагностика: рентгенография органов грудной клетки в двух проекциях "
        "выполняется всем пациентам с подозрением на пневмонию.",
        "page_from": 7,
    },
    {
        "text": "Лечение: амоксициллин 500 мг внутрь три раза в сутки в течение 7 дней.",
        "page_from": 12,
    },
]


def test_exact_substring_match_returns_page() -> None:
    q = "Рентгенография органов грудной клетки в двух проекциях"
    assert locate_page_for_quote(q, _CHUNKS) == 7


def test_match_from_middle_of_quote() -> None:
    q = "Всем пациентам показана рентгенография органов грудной клетки в двух проекциях при подозрении"
    assert locate_page_for_quote(q, _CHUNKS) == 7


def test_token_overlap_fallback() -> None:
    q = "амоксициллин 500 мг внутрь трижды сутки семь дней"
    assert locate_page_for_quote(q, _CHUNKS) == 12


def test_no_match_returns_none() -> None:
    q = "Магнитно-резонансная томография головного мозга с контрастированием гадолинием."
    assert locate_page_for_quote(q, _CHUNKS) is None


def test_short_quote_returns_none() -> None:
    assert locate_page_for_quote("ОАК", _CHUNKS) is None


def test_empty_chunks_returns_none() -> None:
    assert locate_page_for_quote("любой длинный текст цитаты протокола", []) is None
