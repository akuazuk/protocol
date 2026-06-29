#!/usr/bin/env python3
"""Продвинуть rich_chunks.v2.jsonl в основной rich_chunks.jsonl для RAG."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "rich_chunks"
V2 = OUT_DIR / "rich_chunks.v2.jsonl"
FINAL = OUT_DIR / "rich_chunks.final.jsonl"
MAIN = OUT_DIR / "rich_chunks.jsonl"
MANIFEST = OUT_DIR / "_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote rich_chunks.v2.jsonl to production jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=("auto", "v2", "final"), default="auto")
    args = parser.parse_args()

    if args.source == "final":
        src = FINAL
    elif args.source == "v2":
        src = V2
    elif FINAL.is_file():
        src = FINAL
    elif V2.is_file():
        src = V2
    else:
        src = V2

    if not src.is_file():
        print(f"Нет {src}", file=sys.stderr)
        return 1

    n_lines = sum(1 for _ in src.open(encoding="utf-8"))
    if n_lines < 1:
        print(f"REFUSE: пустой источник {src} ({n_lines} chunks)", file=sys.stderr)
        return 1
    indexable_false = 0
    for line in V2.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("indexable") is False:
            indexable_false += 1

    info = {
        "source": str(src),
        "target": str(MAIN),
        "chunks": n_lines,
        "indexable_false": indexable_false,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    if MAIN.is_file():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUT_DIR / f"rich_chunks.pre_v2_{ts}.jsonl"
        shutil.copy2(MAIN, backup)
        print(f"Backup: {backup}")

    shutil.copy2(src, MAIN)

    if MANIFEST.is_file():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            manifest["total_chunks"] = n_lines
            manifest["chunk_quality_v2"] = True
            manifest["indexable_false"] = indexable_false
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"WARN: manifest not updated: {e}", file=sys.stderr)

    print(f"Promoted {n_lines} chunks → {MAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
