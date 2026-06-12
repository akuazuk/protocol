"""Индекс path → лучший текст чанка для обучения embedder."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]

_CHUNK_TYPE_PRIORITY = {
    "title": 0,
    "diagnosis": 1,
    "indications": 2,
    "diagnostics": 3,
    "treatment": 4,
    "body": 5,
}


def _chunks_root() -> Path:
    raw = (os.environ.get("RAG_CHUNKS_DIR") or "").strip()
    base = Path(raw) if raw else ROOT
    one = (os.environ.get("RAG_CHUNKS_JSONL") or "").strip()
    if one:
        return Path(one)
    parts = sorted(base.glob("corpus_chunks_parts/chunks.part.*.jsonl"))
    if parts:
        return base / "corpus_chunks_parts"
    if (base / "corpus_chunks_parts").is_dir():
        return base / "corpus_chunks_parts"
    return base


def iter_chunk_files(root: Path | None = None) -> list[Path]:
    base = root or _chunks_root()
    if base.is_file():
        return [base]
    parts = sorted(base.glob("chunks.part.*.jsonl"))
    if parts:
        return parts
    mini = ROOT / "tests" / "fixtures" / "chunks.mini.jsonl"
    if mini.is_file():
        return [mini]
    return sorted(base.glob("*.jsonl"))


def _text_for_train(row: dict) -> str:
    t = (row.get("embedding_ready_text") or row.get("text") or "").strip()
    return t[:1200]


def _score_chunk(row: dict) -> tuple[int, int]:
    ctype = (row.get("chunk_type") or "body").strip().lower()
    pri = _CHUNK_TYPE_PRIORITY.get(ctype, 9)
    return (pri, -len(_text_for_train(row)))


def iter_corpus_rows(files: list[Path] | None = None) -> Iterator[dict]:
    paths = files or iter_chunk_files()
    for fp in paths:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def build_path_index(
    files: list[Path] | None = None,
    *,
    limit_paths: int = 0,
) -> dict[str, str]:
    """path → лучший passage-текст (приоритет title/diagnosis, до 1200 символов)."""
    best: dict[str, tuple[tuple[int, int], str]] = {}
    for row in iter_corpus_rows(files):
        path = (row.get("source_path") or row.get("path") or "").strip()
        if not path:
            continue
        text = _text_for_train(row)
        if len(text) < 24:
            continue
        key = _score_chunk(row)
        prev = best.get(path)
        if prev is None or key < prev[0]:
            best[path] = (key, text)
        if limit_paths > 0 and len(best) >= limit_paths:
            break
    return {p: v[1] for p, v in best.items()}


def resolve_path_text(path: str, index: dict[str, str]) -> str | None:
    p = (path or "").strip()
    if not p:
        return None
    if p in index:
        return index[p]
    # suffix match (golden path vs corpus path)
    for k, v in index.items():
        if k.endswith(p) or p.endswith(k):
            return v
    base = p.split("/")[-1]
    for k, v in index.items():
        if k.endswith(base) or base in k:
            return v
    return None
