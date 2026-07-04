#!/usr/bin/env python3
"""Дописать chunk_id_global.json к существующему corpus_vector_index без пересборки vectors.npy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.vector_index import _write_chunk_id_sidecar, build_index_from_chunks, load_index


def _jsonl_parts(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(inp.glob("chunks.part.*.jsonl")) or sorted(inp.glob("*.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="corpus_chunks_parts")
    ap.add_argument("--index", default="corpus_vector_index")
    args = ap.parse_args()

    import rag_server as rs

    parts = _jsonl_parts((ROOT / args.input).resolve())
    if not parts:
        print("Нет JSONL", file=sys.stderr)
        return 1
    chunks = rs._load_chunks_from_jsonl(parts)
    index_dir = (ROOT / args.index).resolve()
    meta_path = index_dir / "meta.json"
    if not meta_path.is_file():
        print(f"Нет {meta_path}", file=sys.stderr)
        return 2

    # Загрузить chunk_indices в память модуля vector_index
    loaded = load_index(index_dir)
    if not loaded.get("ok"):
        print(json.dumps(loaded, ensure_ascii=False), file=sys.stderr)
        return 3

    n = _write_chunk_id_sidecar(index_dir, chunks)
    print(json.dumps({"ok": True, "chunk_id_map": n, "path": str(index_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
