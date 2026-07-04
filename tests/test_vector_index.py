"""Векторный индекс (numpy fallback)."""
from __future__ import annotations

from clinical_knowledge.vector_index import build_index_from_chunks, search


def test_vector_search_top_k():
    chunks = [
        {"path": "a.pdf", "text": "one", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"path": "b.pdf", "text": "two", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"path": "c.pdf", "text": "three", "embedding": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    stats = build_index_from_chunks(chunks)
    assert stats["ok"] is True
    hits = search([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=2)
    assert 0 in hits
    assert 2 in hits


def test_search_scoped_with_scores():
    from clinical_knowledge.vector_index import search_scoped_with_scores

    chunks = [
        {"path": "a.pdf", "text": "one", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"path": "b.pdf", "text": "two", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    build_index_from_chunks(chunks)
    hits = search_scoped_with_scores([0.2, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {1}, top_k=1)
    assert hits[0][0] == 1
    assert hits[0][1] > 0.97
