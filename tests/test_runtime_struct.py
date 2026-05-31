"""Рантайм использует структуру корпуса: загрузка чанков сохраняет раздел/страницы/пункты/МКБ,
а контекст consult-review их прокидывает в промпт (за флагом CONSULT_REVIEW_RICH_CONTEXT)."""
from __future__ import annotations

import json


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "chunks.part.000.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_load_chunks_keeps_structure(tmp_path, monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("RAG_KEEP_STRUCT", "1")
    monkeypatch.delenv("RAG_MEMORY_SAVER", raising=False)
    rows = [
        {
            "source_path": "minzdrav_protocols/x/p.pdf",
            "chunk_id": "c1",
            "text": "Диагностика бронхита: рентген и анализ крови.",
            "chunk_type": "diagnostics",
            "section_path": ["2. Диагностика", "2.1. Лабораторная"],
            "section_title": "2.1. Лабораторная",
            "point_numbers": ["2.1"],
            "icd10_codes": ["J20.9"],
            "page_from": 5,
            "page_to": 6,
        }
    ]
    p = _write_jsonl(tmp_path, rows)
    loaded = rs._load_chunks_from_jsonl([p])
    assert len(loaded) == 1
    ch = loaded[0]
    assert ch["section_title"] == "2.1. Лабораторная"
    assert ch["section_path"][-1] == "2.1. Лабораторная"
    assert ch["point_numbers"] == ["2.1"]
    assert ch["icd10_codes"] == ["J20.9"]
    assert ch["page_from"] == 5
    assert ch["page_to"] == 6


def test_load_chunks_struct_can_be_disabled(tmp_path, monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("RAG_KEEP_STRUCT", "0")
    rows = [
        {
            "source_path": "minzdrav_protocols/x/p.pdf",
            "chunk_id": "c1",
            "text": "Текст чанка достаточной длины для индексации.",
            "chunk_type": "body",
            "section_title": "Раздел",
            "section_path": ["Раздел"],
        }
    ]
    p = _write_jsonl(tmp_path, rows)
    loaded = rs._load_chunks_from_jsonl([p])
    assert len(loaded) == 1
    assert "section_title" not in loaded[0]
    assert "section_path" not in loaded[0]


def test_review_context_rich_includes_section_and_pages(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("CONSULT_REVIEW_RICH_CONTEXT", "1")
    rows = [
        {
            "path": "minzdrav_protocols/x/p.pdf",
            "kind": "diagnostics",
            "text": "Выдержка протокола про диагностику.",
            "section_title": "2.1. Лабораторная диагностика",
            "page_from": 5,
            "page_to": 6,
            "point_numbers": ["2.1"],
        }
    ]
    ctx, paths = rs._build_review_chunks_context(rows, 5000)
    assert "section=2.1. Лабораторная диагностика" in ctx
    assert "pages=5-6" in ctx
    assert "пункты=2.1" in ctx
    assert paths == ["minzdrav_protocols/x/p.pdf"]


def test_review_context_plain_when_disabled(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("CONSULT_REVIEW_RICH_CONTEXT", "0")
    rows = [
        {
            "path": "minzdrav_protocols/x/p.pdf",
            "kind": "diagnostics",
            "text": "Выдержка протокола.",
            "section_title": "Раздел",
            "page_from": 5,
            "page_to": 6,
        }
    ]
    ctx, _ = rs._build_review_chunks_context(rows, 5000)
    assert "section=" not in ctx
    assert "pages=" not in ctx


def test_consult_rag_second_pass_default_off_on_render(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.delenv("CONSULT_REVIEW_RAG_SECOND_PASS", raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert rs._consult_rag_second_pass_enabled() is False


def test_consult_rag_second_pass_default_on_locally(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.delenv("CONSULT_REVIEW_RAG_SECOND_PASS", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    assert rs._consult_rag_second_pass_enabled() is True


def test_consult_rag_second_pass_env_overrides_render(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("CONSULT_REVIEW_RAG_SECOND_PASS", "1")
    assert rs._consult_rag_second_pass_enabled() is True
