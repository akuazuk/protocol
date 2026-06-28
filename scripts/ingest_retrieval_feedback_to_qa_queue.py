#!/usr/bin/env python3
"""Добавить source_path из feedback в chunk_qa_queue."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = ROOT / "data" / "ml" / "feedback"
DEFAULT_QUEUE = ROOT / "data" / "ml" / "chunk_qa_queue.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest retrieval feedback into QA queue")
    parser.add_argument("--out", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()

    existing: set[str] = set()
    rows: list[dict] = []
    if args.out.is_file():
        for line in args.out.open(encoding="utf-8"):
            row = json.loads(line)
            rows.append(row)
            existing.add(str(row.get("chunk_id") or ""))

    added = 0
    if FEEDBACK_DIR.is_dir():
        for fp in FEEDBACK_DIR.glob("*.jsonl"):
            for line in fp.open(encoding="utf-8"):
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp = ev.get("source_path") or ev.get("path")
                if not sp:
                    continue
                cid = f"feedback:{sp}"
                if cid in existing:
                    continue
                rows.append({
                    "chunk_id": cid,
                    "source_path": sp,
                    "priority": 100,
                    "issues": ["retrieval_feedback"],
                    "from_feedback": str(fp.name),
                })
                existing.add(cid)
                added += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"added": added, "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
