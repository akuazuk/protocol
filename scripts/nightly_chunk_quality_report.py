#!/usr/bin/env python3
"""Nightly отчёт качества чанков."""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "data" / "ml" / "reports" / "chunk_quality_baseline.json"
CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
FALLBACK = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"


def main() -> int:
    chunks = CHUNKS if CHUNKS.is_file() else FALLBACK
    report = ROOT / "data" / "ml" / "reports" / f"chunk_quality_nightly_{date.today().isoformat()}.md"
    stats = ROOT / "data" / "ml" / "reports" / f"chunk_quality_nightly_{date.today().isoformat()}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "audit_chunk_quality.py"),
        "--chunks", str(chunks),
        "--report", str(report),
        "--stats", str(stats),
    ]
    if BASELINE.is_file():
        cmd.extend(["--baseline", str(BASELINE)])
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
