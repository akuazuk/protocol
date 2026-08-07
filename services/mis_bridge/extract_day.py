#!/usr/bin/env python3
"""Thin CLI: document extract contract paths (wrapper era E1).

Полный SQL extract по-прежнему в scripts/export_mis_* / mo daily.
Этот модуль фиксирует OUT_DIR layout для будущих upload-ов.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=date.fromisoformat, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: $MO_DATA_ROOT/inbound/extract",
    )
    ap.add_argument(
        "--run-host",
        default=os.environ.get("RUN_HOST", "mac"),
        choices=("mac", "gcp", "by"),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="только создать meta-шаблон и напечатать пути",
    )
    args = ap.parse_args(argv)

    root = Path(os.environ.get("MO_DATA_ROOT", "data/medical_exams")).expanduser()
    out_dir = args.out_dir or (root / "inbound" / "extract")
    out_dir.mkdir(parents=True, exist_ok=True)
    day = args.day.isoformat()
    csv_path = out_dir / f"mo_{day}.csv"
    meta_path = out_dir / f"mo_{day}.meta.json"
    meta = {
        "schema_version": 1,
        "day": day,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "run_host": args.run_host,
        "row_count": 0,
        "checksum_sha256": "",
        "source": "kravira_mc.mis_protocol",
        "csv_path": str(csv_path),
        "note": "wrapper: fill csv via existing export_mis / mo daily extract",
    }
    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")
    print(f"expected_csv {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
