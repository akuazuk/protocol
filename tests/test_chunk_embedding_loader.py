"""Загрузка embedding из JSONL в rag_server."""
from __future__ import annotations

import json
import os
from pathlib import Path

import rag_server as rs


def test_load_chunks_preserves_embedding(tmp_path: Path, monkeypatch):
    row = {
        "chunk_id": "t1",
        "source_path": "minzdrav_protocols/x/a.pdf",
        "text": "тестовый чанк",
        "page_from": 1,
        "page_to": 1,
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "embedding_model": "test-model",
        "embedding_dim": 8,
    }
    p = tmp_path / "mini.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.delenv("RAG_MEMORY_SAVER", raising=False)
    chunks = rs._load_chunks_from_jsonl([p])
    assert len(chunks) == 1
    assert chunks[0]["embedding"] == row["embedding"]
    assert chunks[0]["embedding_model"] == "test-model"
