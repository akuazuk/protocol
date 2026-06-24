#!/usr/bin/env python3
"""Манифест корпуса: sha256 частей, число чанков, наличие embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(corpus_dir: Path) -> dict:
    parts = sorted(corpus_dir.glob("chunks.part.*.jsonl"))
    if not parts:
        parts = sorted(corpus_dir.glob("*.jsonl"))
    files_meta = []
    total_chunks = 0
    with_embedding = 0
    for p in parts:
        n = 0
        emb = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    row = json.loads(line)
                    if isinstance(row.get("embedding"), list):
                        emb += 1
                except json.JSONDecodeError:
                    pass
        total_chunks += n
        with_embedding += emb
        files_meta.append(
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
                "chunks": n,
                "chunks_with_embedding": emb,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "parts_count": len(files_meta),
        "total_chunks": total_chunks,
        "chunks_with_embedding": with_embedding,
        "embedding_coverage_pct": round(100.0 * with_embedding / total_chunks, 2) if total_chunks else 0.0,
        "files": files_meta,
        "path_manifest_note": "Для lazy load по PDF: scripts/build_corpus_path_manifest.py → corpus_path_manifest.jsonl",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus_chunks_parts")
    parser.add_argument("--output", default="output/corpus_manifest.json")
    args = parser.parse_args()
    corpus = (ROOT / args.corpus).resolve()
    if not corpus.is_dir():
        print(f"Нет каталога {corpus}", file=sys.stderr)
        return 1
    manifest = build_manifest(corpus)
    out = (ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
