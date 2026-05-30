"""Разметка корпуса: иерархия разделов, валидация МКБ, склейка многостраничных таблиц."""
from __future__ import annotations


def test_section_hierarchy_and_numbering() -> None:
    from corpus_pipeline.section_detect import detect_sections

    text = (
        "1. ОБЩИЕ ПОЛОЖЕНИЯ\n"
        "текст раздела достаточно длинный для прохождения порога минимум.\n"
        "2. ДИАГНОСТИКА\n"
        "клиническая картина включает кашель и лихорадку у пациента.\n"
        "2.1. Лабораторная диагностика\n"
        "анализ крови и мокроты обязателен для подтверждения диагноза.\n"
    )
    secs = detect_sections("doc1", text)
    by_num = {s["section_number"]: s for s in secs}
    assert "2" in by_num and "2.1" in by_num
    # Подраздел 2.1 вложен под 2.
    sub = by_num["2.1"]
    assert sub["section_path"][0] == by_num["2"]["section_title"]
    assert sub["section_path"][-1] == sub["section_title"]


def test_icd_validation_filters_invalid_roots() -> None:
    from corpus_pipeline.entities_extract import extract_icd10, _valid_icd_roots

    # Справочник должен загрузиться (иначе тест бессмысленен).
    assert len(_valid_icd_roots()) > 100
    codes = extract_icd10("Диагноз J20.9, также J45.0 и явно несуществующий Q99.9? plus Z00.0")
    assert "J20.9" in codes
    assert "J45.0" in codes
    # Невалидный корень не проходит.
    assert all(c[:3] in _valid_icd_roots() for c in codes)


def test_merge_multipage_tables_same_header() -> None:
    from corpus_pipeline.tables_extract import merge_multipage_tables

    tables = [
        {
            "page": 1,
            "table_index_on_page": 0,
            "columns": ["Препарат", "Доза"],
            "rows": [["A", "10 мг"]],
            "raw_markdown": "x",
        },
        {
            "page": 2,
            "table_index_on_page": 0,
            "columns": ["Препарат", "Доза"],
            "rows": [["B", "20 мг"]],
            "raw_markdown": "y",
        },
    ]
    merged = merge_multipage_tables(tables)
    assert len(merged) == 1
    assert merged[0]["page_from"] == 1
    assert merged[0]["page_to"] == 2
    assert len(merged[0]["rows"]) == 2


def test_merge_multipage_tables_distinct_kept() -> None:
    from corpus_pipeline.tables_extract import merge_multipage_tables

    tables = [
        {"page": 1, "table_index_on_page": 0, "columns": ["A", "B"], "rows": [["1", "2"]], "raw_markdown": "x"},
        {"page": 5, "table_index_on_page": 0, "columns": ["X", "Y", "Z"], "rows": [["3", "4", "5"]], "raw_markdown": "y"},
    ]
    merged = merge_multipage_tables(tables)
    assert len(merged) == 2
