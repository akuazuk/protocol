"""Семантический поиск и AI Overview по протоколу."""
from __future__ import annotations

from clinical_knowledge.protocol_semantic_search import (
    _merge_score,
    _parse_overview_json,
    detect_query_intents,
)
from clinical_knowledge.protocol_source_view import format_rich_chunk_nav_item
from clinical_knowledge.vector_index import build_index_from_chunks, search_scoped_with_scores


def _treatment_chunk() -> dict:
    return {
        "path": "minzdrav_protocols/test.pdf",
        "chunk_type": "treatment",
        "text": (
            "Фармакотерапия включает диосмин, гесперидин и компрессионную терапию. "
            "При варикозном расширении вен назначают ФЛП и склеротерапию."
        ),
        "page_from": 12,
        "drugs": ["диосмин", "гесперидин"],
        "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _diagnostics_chunk() -> dict:
    return {
        "path": "minzdrav_protocols/test.pdf",
        "chunk_type": "diagnostics",
        "text": "Диагностика включает УЗДС вен нижних конечностей и лабораторные анализы крови.",
        "page_from": 8,
        "imaging": ["УЗДС"],
        "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_detect_query_intents_drugs():
    intents = detect_query_intents("какие лекарства назначить при варикозе")
    assert "treatment" in intents


def test_detect_query_intents_diagnostics():
    intents = detect_query_intents("какие обследования сдать")
    assert "diagnostics" in intents


def test_merge_score_weights():
    score = _merge_score(cosine=0.9, lex=0.2, intent=0.5)
    assert 0.5 < score < 1.0


def test_format_rich_chunk_nav_item():
    item = format_rich_chunk_nav_item(_treatment_chunk())
    assert item is not None
    assert item["section_id"] == "treatment"
    assert "диосмин" in (item.get("search_blob") or "")


def test_search_scoped_with_scores():
    chunks = [_treatment_chunk(), _diagnostics_chunk()]
    stats = build_index_from_chunks(chunks)
    assert stats["ok"] is True
    hits = search_scoped_with_scores(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        {0},
        top_k=2,
    )
    assert hits == [(0, 1.0)]


def test_parse_overview_json():
    raw = '{"summary":"Кратко.","points":[{"text":"Назначить ФЛП","source_idx":0}]}'
    parsed = _parse_overview_json(raw)
    assert parsed is not None
    assert parsed["summary"] == "Кратко."
    assert len(parsed["points"]) == 1
