#!/usr/bin/env python3
"""Orchestrate LLM summary pipeline phases A-D."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    rc = 0

    if not args.skip_source:
        cmd = [py, "scripts/prepare_protocol_source_text.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        rc = _run(cmd)
        if rc:
            return rc

    if not args.skip_batch:
        cmd = [py, "scripts/batch_llm_protocol_summaries.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.resume:
            cmd.append("--resume")
        if args.no_llm:
            cmd.append("--no-llm")
        rc = _run(cmd)
        if rc:
            return rc

    for script in (
        "scripts/validate_protocol_summaries.py",
        "scripts/export_protocol_summary_rag.py",
        "scripts/build_protocol_catalog.py",
        "scripts/build_protocol_icd_index.py",
    ):
        rc = _run([py, script])
        if rc:
            return rc
    print("Pipeline A-D complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
