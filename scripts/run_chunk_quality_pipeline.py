#!/usr/bin/env python3
"""Оркестратор: enrich → apply v2 → audit → queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print(f"FAIL ({rc}): {' '.join(cmd)}", file=sys.stderr)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full chunk quality pipeline")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-apply", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Copy v2 → rich_chunks.jsonl after apply")
    parser.add_argument("--queue-max", type=int, default=12000)
    args = parser.parse_args()

    chunks_v1 = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
    chunks_v2 = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
    report_dir = ROOT / "data" / "ml" / "reports"
    today = date.today().isoformat()

    if not args.skip_enrich:
        if _run([PY, "scripts/enrich_rich_chunk_tags.py"], "enrich tags") != 0:
            return 1

    if not args.skip_apply:
        if _run([PY, "scripts/apply_chunk_rule_fixes.py"], "apply rule fixes") != 0:
            return 1

    baseline = report_dir / "chunk_quality_baseline.json"
    if not baseline.is_file() and chunks_v1.is_file():
        _run([
            PY, "scripts/audit_chunk_quality.py",
            "--chunks", str(chunks_v1),
            "--stats", str(baseline),
            "--report", str(report_dir / "chunk_quality_baseline.md"),
        ], "baseline audit (one-time)")

    audit_stats = report_dir / f"chunk_quality_{today}.json"
    cmd = [
        PY, "scripts/audit_chunk_quality.py",
        "--chunks", str(chunks_v2 if chunks_v2.is_file() else chunks_v1),
        "--stats", str(audit_stats),
        "--report", str(report_dir / f"chunk_quality_{today}.md"),
    ]
    if baseline.is_file():
        cmd.extend(["--baseline", str(baseline)])
    if _run(cmd, "audit v2") != 0:
        return 1

    _run([
        PY, "scripts/build_chunk_qa_queue.py",
        "--max", str(args.queue_max),
    ], "build LLM QA queue")

    if args.promote and chunks_v2.is_file():
        if _run([PY, "scripts/promote_rich_chunks_v2.py"], "promote v2") != 0:
            return 1

    summary = {
        "v2_exists": chunks_v2.is_file(),
        "promoted": args.promote,
        "audit_stats": str(audit_stats),
    }
    if audit_stats.is_file():
        summary["metrics"] = json.loads(audit_stats.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
