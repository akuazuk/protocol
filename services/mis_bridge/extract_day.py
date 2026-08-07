#!/usr/bin/env python3
"""Prepare day extract for Mac→GCS→GCP (extract-contract layout).

Modes:
  --from-csv PATH   package existing day CSV (secure_cases or staging)
  --from-secure     auto: $MO_DATA_ROOT/secure_cases/YYYY/MM/mo_DAY.csv
  (default)         same as --from-secure if CSV exists, else write meta stub only

Does not call MariaDB itself (VPN window stays in export_mis / mo daily).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(csv_path: Path) -> int:
    # header + data rows; empty file → 0
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-csv", type=Path, default=None, help="source day CSV")
    src.add_argument(
        "--from-secure",
        action="store_true",
        help="use secure_cases/YYYY/MM/mo_DAY.csv under MO_DATA_ROOT",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned paths/meta, do not write",
    )
    args = ap.parse_args(argv)

    root = Path(os.environ.get("MO_DATA_ROOT", "data/medical_exams")).expanduser()
    out_dir = args.out_dir or (root / "inbound" / "extract")
    day = args.day.isoformat()
    csv_out = out_dir / f"mo_{day}.csv"
    meta_out = out_dir / f"mo_{day}.meta.json"

    source_csv: Path | None = None
    if args.from_csv is not None:
        source_csv = args.from_csv.expanduser()
    else:
        # --from-secure or default auto
        candidate = root / "secure_cases" / f"{args.day:%Y}" / f"{args.day:%m}" / f"mo_{day}.csv"
        if args.from_secure or candidate.is_file():
            source_csv = candidate

    meta: dict = {
        "schema_version": 1,
        "day": day,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "run_host": args.run_host,
        "row_count": 0,
        "checksum_sha256": "",
        "source": "kravira_mc.mis_protocol",
        "csv_path": str(csv_out),
    }

    if source_csv is None:
        meta["note"] = "no source csv; meta stub only (run MIS export or pass --from-csv)"
        if args.dry_run:
            print(json.dumps(meta, ensure_ascii=False, indent=2))
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {meta_out}")
        print(f"expected_csv {csv_out}")
        return 0

    if not source_csv.is_file():
        print(f"ERROR: source csv missing: {source_csv}", flush=True)
        return 2

    meta["source_csv"] = str(source_csv)
    meta["row_count"] = _row_count(source_csv)
    meta["checksum_sha256"] = _sha256(source_csv)
    meta["note"] = "packaged for GCS inbound/extract (B4)"

    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if source_csv.resolve() != csv_out.resolve():
        shutil.copy2(source_csv, csv_out)
    # recompute checksum of destination (same content)
    meta["checksum_sha256"] = _sha256(csv_out)
    meta["csv_path"] = str(csv_out)
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_out} rows={meta['row_count']} sha256={meta['checksum_sha256'][:12]}…")
    print(f"wrote {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
