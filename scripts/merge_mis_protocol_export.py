#!/usr/bin/env python3
"""Идемпотентно пересобрать rolling month из daily Parquet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import merge_daily_partitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.daily_dir.glob(f"mo_{args.month}-*.parquet"))
    if not paths:
        parser.error("daily partitions не найдены")
    parquet, csv_path, details = merge_daily_partitions(paths, month=args.month, out_dir=args.out_dir)
    print(json.dumps({**details, "parquet": str(parquet), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
