"""Lite практический разбор из rich-чанков."""
from __future__ import annotations

from clinical_knowledge.protocol_practical_lite import (
    build_clinical_detail_lite,
    build_lite_sections,
)


def _chunk(text: str, ctype: str = "diagnostics", **kw) -> dict:
    return {
        "rich_chunk": True,
        "text": text,
        "chunk_type": ctype,
        "page_from": 3,
        **kw,
    }


def test_lite_sections_from_rich_chunks():
    chunks = [
        _chunk("Рентгенография органов грудной клетки при подозрении на бронхит", "diagnostics"),
        _chunk("Амоксициллин 500 мг 3 раза в сутки 5-7 дней", "pharmacotherapy"),
    ]
    secs = build_lite_sections(chunks, "J20.9 острый бронхит кашель", ["J20.9"])
    assert len(secs) >= 1
    assert secs[0].get("text")


def test_clinical_detail_lite_has_extraction_lists():
    chunks = [
        _chunk(
            "Обязательно: общий анализ крови\nРентгенография при необходимости",
            "diagnostics",
        ),
        _chunk("Ингаляционные бронхолитики при обструкции", "treatment"),
    ]
    cd = build_clinical_detail_lite(
        "minzdrav_protocols/x/kp.pdf",
        "острый бронхит",
        "КП бронхит",
        chunks,
        ["J20.9"],
    )
    assert cd["source"] == "rich_chunks"
    assert cd["llm_used"] is False
    ex = cd["extraction"]
    assert ex.get("investigations") or ex.get("treatment_methods")
    assert cd.get("lite_sections")


def test_build_practical_section_medications():
    from clinical_knowledge.protocol_practical_lite import build_practical_section

    chunks = [
        _chunk("Рентгенография органов грудной клетки", "diagnostics"),
        _chunk("Амоксициллин 500 мг 3 раза в сутки 5-7 дней", "pharmacotherapy"),
    ]
    out = build_practical_section(
        "minzdrav_protocols/x/kp.pdf",
        "острый бронхит",
        "КП бронхит",
        chunks,
        "medications",
        ["J20.9"],
    )
    assert out["section"] == "medications"
    assert out["items"]
    assert any("Амоксициллин" in it for it in out["items"])
