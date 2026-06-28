"""Тесты rich-chunk search helpers."""
from __future__ import annotations

from clinical_knowledge.rich_chunk_search import (
    build_chunk_match_reason,
    build_rich_protocol_nav,
    chunk_type_multiplier,
    detect_query_intent,
    hybrid_merge_protocols,
    hybrid_pin_trusted_icd_top1,
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


def test_should_skip_short_non_overview_chunk():
    row = {
        "doc_id": "abc",
        "chunk_type": "body",
        "text": "Короткий фрагмент без смысла.",
    }
    assert should_skip_rich_chunk_row(row) is True


def test_should_skip_indexable_false():
    row = {
        "doc_id": "abc",
        "chunk_type": "diagnostics",
        "text": "Рекомендуется выполнить ОАК при постановке диагноза антифосфолипидного синдрома.",
        "indexable": False,
    }
    assert should_skip_rich_chunk_row(row) is True


def test_should_skip_terms_without_icd():
    row = {
        "doc_id": "abc",
        "chunk_type": "terms",
        "text": "Термины и определения используемые в настоящем клиническом протоколе медицинской помощи.",
    }
    assert should_skip_rich_chunk_row(row) is True


def test_hybrid_pin_trusted_icd_top1():
    icd = [{"path": "minzdrav_protocols/a/orvi.pdf", "title": "ОРВИ", "confidence_score": 0.88}]
    rag = [
        {"path": "minzdrav_protocols/a/copd.pdf", "title": "ХОБЛ", "confidence_score": 0.95},
        {"path": "minzdrav_protocols/a/orvi.pdf", "title": "ОРВИ", "confidence_score": 0.7},
    ]
    merged = hybrid_merge_protocols(icd, rag, icd_weight=0.62, rag_weight=0.38)
    assert merged[0]["path"].endswith("copd.pdf")
    pinned = hybrid_pin_trusted_icd_top1(
        merged,
        icd,
        query="кашель и температура 38",
        ambiguous=False,
        icd_codes=["J06.9"],
    )
    assert pinned[0]["path"].endswith("orvi.pdf")
    assert pinned[0].get("hybrid_icd_pinned") is True


def test_chunk_type_multiplier_treatment_intent():
    ch = {"rich_chunk": True, "chunk_type": "treatment", "kind": "treatment", "text": "назначают терапию"}
    mult = chunk_type_multiplier("лечение антибиотиками", ch)
    assert mult > 1.0


def test_chunk_type_multiplier_overview_icd_overlap():
    ch = {
        "rich_chunk": True,
        "chunk_type": "protocol_overview",
        "kind": "protocol_overview",
        "icd10_weights": {"J06.9": 95},
    }
    mult = chunk_type_multiplier("кашель", ch, icd_codes=["J06.9"])
    assert mult >= 2.0


def test_chunk_type_multiplier_terms_low():
    ch = {"rich_chunk": True, "chunk_type": "terms", "kind": "terms", "text": "термины"}
    mult = chunk_type_multiplier("кашель", ch)
    assert mult <= 0.4


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
