#!/usr/bin/env python3
"""Phase A: build source_text/{protocol_id}.json for all catalog PDFs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.source_fingerprint import catalog_sha256, save_fingerprint  # noqa: E402
from clinical_knowledge.protocol_summary.source_text import (  # noqa: E402
    CATALOG,
    build_source_text_document,
    save_source_text,
)


def _catalog_paths(limit: int | None = None) -> list[str]:
    paths: list[str] = []
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = str(row.get("path") or "").replace("\\", "/")
            if p:
                paths.append(p)
    paths = sorted(set(paths))
    if limit:
        paths = paths[:limit]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--path", type=str, default=None, help="single PDF path from catalog")
    args = ap.parse_args()
    paths = [args.path.replace("\\", "/")] if args.path else _catalog_paths(args.limit)
    done = 0
    for path in paths:
        doc = build_source_text_document(path)
        out = save_source_text(doc)
        sha = catalog_sha256(path)
        if sha:
            save_fingerprint(str(doc.get("protocol_id") or ""), sha, path)
        done += 1
        print(f"OK {doc.get('protocol_id')} -> {out.name} chunks={doc.get('chunk_count')}")
    print(f"Prepared {done} source_text documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
