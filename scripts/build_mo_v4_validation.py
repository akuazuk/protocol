#!/usr/bin/env python3
"""Build the secure 300-case double-label queue and report validation gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_validation import (
    build_gold_queue,
    evaluate_gold,
    protocol_trust_status,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--protocols",
        type=Path,
        default=ROOT / "data" / "protocol_summaries" / "json",
    )
    args = parser.parse_args()
    if args.queue.is_file():
        queue = load_jsonl(args.queue)
    else:
        queue = build_gold_queue(load_jsonl(args.cases), size=300)
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        args.queue.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
            encoding="utf-8",
        )
    report = {
        "gold": evaluate_gold(queue),
        "protocol_trust": protocol_trust_status(args.protocols),
        "requirements": {
            "gold_n": 300,
            "spearman": 0.70,
            "p0_recall": 0.90,
            "penalty_ready_protocols": 120,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
