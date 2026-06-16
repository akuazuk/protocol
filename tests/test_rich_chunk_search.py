"""Тесты rich-chunk search helpers."""
from __future__ import annotations

from clinical_knowledge.rich_chunk_search import (
    build_chunk_match_reason,
    build_rich_protocol_nav,
    chunk_type_multiplier,
    detect_query_intent,
    hybrid_merge_protocols,
    should_skip_rich_chunk_row,
)


def test_should_skip_preamble_chunk():
    row = {
        "doc_id": "abc",
        "chunk_type": "body",
        "text": "ПОСТАНОВЛЕНИЕ МИНИСТЕРСТВА ЗДРАВООХРАНЕНИЯ\nОб утверждении клинического протокола",
        "is_preamble_filtered": True,
    }
    assert should_skip_rich_chunk_row(row) is True


def test_chunk_type_multiplier_treatment_intent():
    ch = {"rich_chunk": True, "chunk_type": "treatment", "kind": "treatment", "text": "назначают терапию"}
    mult = chunk_type_multiplier("лечение антибиотиками", ch)
    assert mult > 1.0


def test_hybrid_merge_prefers_both_signals():
    icd = [{"path": "minzdrav_protocols/a/foo.pdf", "title": "A", "confidence_score": 0.9}]
    rag = [{"path": "minzdrav_protocols/a/foo.pdf", "title": "A", "confidence_score": 0.7, "match_reason": "Лечение, стр. 5"}]
    merged = hybrid_merge_protocols(icd, rag, icd_weight=0.4, rag_weight=0.6)
    assert len(merged) == 1
    assert merged[0]["confidence_score"] > 0.7
    assert "Лечение" in (merged[0].get("match_reason") or "")


def test_build_match_reason_with_section():
    row = {
        "kind": "diagnostics",
        "section_title": "1. Диагностика",
        "page_from": 12,
        "icd10_codes": ["J06.9"],
    }
    reason = build_chunk_match_reason(row, ["J06.9"])
    assert "Диагностика" in reason or "МКБ" in reason


def test_rich_protocol_nav_from_chunks():
    chunks = [
        {
            "kind": "diagnostics",
            "chunk_type": "diagnostics",
            "section_title": "Диагностика",
            "text": "Показания к обследованию пациента с ОРВИ.",
            "page_from": 3,
            "chunk_id": "c1",
        },
        {
            "kind": "treatment",
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "text": "Назначают симптоматическую терапию.",
            "page_from": 8,
            "chunk_id": "c2",
        },
    ]
    nav = build_rich_protocol_nav(chunks, path="minzdrav_protocols/x/a.pdf", query="кашель")
    assert nav["available"] is True
    assert nav["source"] == "rich_chunks"
    assert len(nav["conditions"][0]["sections"]) >= 2


if __name__ == "__main__":
    test_should_skip_preamble_chunk()
    test_chunk_type_multiplier_treatment_intent()
    test_hybrid_merge_prefers_both_signals()
    test_build_match_reason_with_section()
    test_rich_protocol_nav_from_chunks()
    print("ok")
