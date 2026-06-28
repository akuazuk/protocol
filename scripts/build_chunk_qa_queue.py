#!/usr/bin/env python3
"""Построить очередь чанков для LLM-QA."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import detect_issues, quality_score

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
FALLBACK_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
DEFAULT_QUEUE = ROOT / "data" / "ml" / "chunk_qa_queue.jsonl"
FEEDBACK_DIR = ROOT / "data" / "ml" / "feedback"

_CLINICAL_TYPES = frozenset({
    "diagnostics", "treatment", "criteria_block", "pharmacotherapy", "drug_list",
})


def _feedback_paths() -> set[str]:
    out: set[str] = set()
    if not FEEDBACK_DIR.is_dir():
        return out
    for fp in FEEDBACK_DIR.glob("*.jsonl"):
        try:
            for line in fp.open(encoding="utf-8"):
                row = json.loads(line)
                sp = row.get("source_path") or row.get("path") or row.get("protocol_path")
                if sp:
                    out.add(str(sp))
        except Exception:
            continue
    return out


def build_queue(
    chunks_path: Path,
    *,
    max_items: int = 12000,
    random_pct: float = 0.02,
    score_threshold: float = 0.5,
) -> list[dict]:
    feedback_paths = _feedback_paths()
    queued: list[tuple[int, dict]] = []
    all_rows: list[dict] = []

    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            all_rows.append(row)
            cid = str(row.get("chunk_id") or "")
            score = quality_score(row)
            issues = detect_issues(row)
            priority = 0
            sp = str(row.get("source_path") or "")
            if sp in feedback_paths:
                priority = 100
            elif score < score_threshold:
                priority = 80
            elif "type_body_but_clinical" in issues:
                priority = 70
            elif "weak_section_title" in issues:
                priority = 60
            elif (row.get("chunk_type") or "") in _CLINICAL_TYPES:
                priority = 50
            elif score < 0.7:
                priority = 40
            if priority > 0:
                queued.append((priority, {
                    "chunk_id": cid,
                    "doc_id": row.get("doc_id"),
                    "source_path": sp,
                    "chunk_type": row.get("chunk_type"),
                    "quality_score": score,
                    "issues": issues,
                    "priority": priority,
                }))

    queued.sort(key=lambda x: (-x[0], x[1].get("chunk_id", "")))

    # random sample for calibration
    n_random = max(1, int(len(all_rows) * random_pct))
    random.seed(42)
    for row in random.sample(all_rows, min(n_random, len(all_rows))):
        cid = str(row.get("chunk_id") or "")
        if any(q[1]["chunk_id"] == cid for q in queued):
            continue
        queued.append((10, {
            "chunk_id": cid,
            "doc_id": row.get("doc_id"),
            "source_path": row.get("source_path"),
            "chunk_type": row.get("chunk_type"),
            "quality_score": quality_score(row),
            "issues": detect_issues(row),
            "priority": 10,
            "sample": True,
        }))

    queued.sort(key=lambda x: (-x[0], x[1].get("chunk_id", "")))
    return [item for _, item in queued[:max_items]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LLM chunk QA queue")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--max", type=int, default=12000)
    parser.add_argument("--random-pct", type=float, default=0.02)
    args = parser.parse_args()

    chunks_path = args.chunks
    if chunks_path is None:
        chunks_path = DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK_CHUNKS
    if not chunks_path.is_file():
        print(f"Нет файла чанков: {chunks_path}", file=sys.stderr)
        return 1

    queue = build_queue(chunks_path, max_items=args.max, random_pct=args.random_pct)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for item in queue:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"queue_size": len(queue), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
