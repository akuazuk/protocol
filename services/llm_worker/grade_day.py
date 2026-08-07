#!/usr/bin/env python3
"""Thin CLI around scripts/grade_kz_llm.py using llm outbox/inbox layout."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=date.fromisoformat, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("MO_DATA_ROOT", "data/medical_exams")),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="проверить layout/manifest, не звать Gemini",
    )
    args = ap.parse_args(argv)

    data_root = args.data_root.expanduser()
    outbox = data_root / "llm_outbox" / args.run_id
    inbox = data_root / "llm_inbox" / args.run_id
    manifest_path = outbox / "manifest.json"
    cases_path = outbox / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        print(f"missing outbox files under {outbox}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    day = args.day.isoformat()
    if str(manifest.get("day") or "") not in ("", day):
        print("manifest.day mismatch", file=sys.stderr)
        return 2

    inbox.mkdir(parents=True, exist_ok=True)
    grades_out = inbox / f"kz_l1_{day}_llm_grades.jsonl"
    result_manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "day": day,
        "grades_ok": 0,
        "grades_err": 0,
        "judges_ok": 0,
        "finished_at": "",
        "model_primary": "dry-run" if args.dry_run else "gemini",
    }

    if args.dry_run:
        (inbox / "result_manifest.json").write_text(
            json.dumps(result_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "dry_run": True, "inbox": str(inbox)}, ensure_ascii=False))
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "grade_kz_llm.py"),
        "--cases",
        str(cases_path),
        "--out",
        str(grades_out),
    ]
    if manifest.get("escalate"):
        cmd.append("--escalate")
    queue = outbox / "llm_queue.json"
    if queue.is_file():
        cmd.extend(["--queue", str(queue)])
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        return proc.returncode

    ok = 0
    if grades_out.is_file():
        ok = sum(1 for line in grades_out.read_text(encoding="utf-8").splitlines() if line.strip())
    result_manifest["grades_ok"] = ok
    from datetime import datetime, timezone

    result_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (inbox / "result_manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
