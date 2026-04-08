"""Pytest: до импорта rag_server задаём минимальный корпус (tests/fixtures/chunks.mini.jsonl)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MINI_JSONL = _ROOT / "tests" / "fixtures" / "chunks.mini.jsonl"

# Изолированный корпус для CI и локального pytest без полного corpus_chunks_parts
if _MINI_JSONL.is_file():
    os.environ.setdefault("RAG_CHUNKS_JSONL", str(_MINI_JSONL))
    os.environ.setdefault("RAG_CHUNKS_SOURCE", "jsonl")

# Без вызова Gemini embedding в retrieve (стабильные юнит-тесты без ключа API)
os.environ.setdefault("RAG_GEMINI_EMBED_RERANK", "0")
