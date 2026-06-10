#!/usr/bin/env python3
"""Построить FAISS/numpy индекс из JSONL с полем embedding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.vector_index import save_index


def _load_chunks(paths: list[Path]) -> list[dict]:
    import rag_server as rs

    return rs._load_chunks_from_jsonl(paths)


def _jsonl_parts(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(inp.glob("chunks.part.*.jsonl")) or sorted(inp.glob("*.jsonl"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="corpus_chunks_parts")
    parser.add_argument("--output", default="corpus_vector_index")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    inp = (ROOT / args.input).resolve()
    out = (ROOT / args.output).resolve()
    parts = _jsonl_parts(inp)
    if not parts:
        print(f"Нет JSONL: {inp}", file=sys.stderr)
        return 1
    chunks = _load_chunks(parts)
    with_emb = sum(1 for c in chunks if c.get("embedding"))
    if with_emb == 0:
        print(
            "Нет embedding в чанках. Сначала: python3 scripts/build_chunk_embeddings.py",
            file=sys.stderr,
        )
        return 2
    stats = save_index(out, chunks, model=args.model)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
