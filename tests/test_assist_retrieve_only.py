"""Тест быстрого assist без LLM (retrieve_only)."""
from __future__ import annotations

from rag_server import _build_protocols_from_retrieval, dedupe_retrieval_by_basename


def test_build_protocols_from_retrieval_orders_by_score():
    rows = [
        {"path": "pulmonologiya/a.pdf", "score": 10.0, "excerpt": "x"},
        {"path": "pulmonologiya/b.pdf", "score": 5.0, "excerpt": "y"},
    ]
    protos = _build_protocols_from_retrieval(rows)
    assert len(protos) == 2
    assert protos[0]["path"].endswith("a.pdf")
    assert float(protos[0]["confidence_score"]) > float(protos[1]["confidence_score"])


def test_dedupe_retrieval_by_basename_one_per_pdf():
    rows = [
        {"path": "rubric1/same.pdf", "score": 8.0},
        {"path": "rubric2/same.pdf", "score": 9.0},
    ]
    out = dedupe_retrieval_by_basename(rows)
    assert len(out) == 1
