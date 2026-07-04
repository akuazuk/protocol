"""Векторный индекс (numpy / mmap fallback)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clinical_knowledge.vector_index import (
    build_index_from_chunks,
    index_stats,
    load_index,
    search,
    search_scoped_with_scores,
)


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
    chunks = [
        {"path": "a.pdf", "text": "one", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"path": "b.pdf", "text": "two", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    build_index_from_chunks(chunks)
    hits = search_scoped_with_scores([0.2, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {1}, top_k=1)
    assert hits[0][0] == 1
    assert hits[0][1] > 0.97


def test_load_index_mmap(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_MMAP", "1")
    chunks = [
        {"path": "a.pdf", "text": "one", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"path": "b.pdf", "text": "two", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    build_index_from_chunks(chunks)
    from clinical_knowledge import vector_index as vi

    assert vi._index_vectors is not None
    np.save(str(tmp_path / "vectors.npy"), np.asarray(vi._index_vectors, dtype=np.float32))
    meta = {
        "chunk_indices": vi._index_chunk_indices,
        "dim": vi._index_dim,
        "count": len(vi._index_chunk_indices or []),
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    vi._index_vectors = None
    vi._index_chunk_indices = None
    vi._faiss_index = None

    stats = load_index(tmp_path)
    assert stats["ok"] is True
    assert stats.get("mmap") is True
    assert index_stats()["backend"] == "numpy_mmap"
    hits = search([0.1, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=1)
    assert 1 in hits
