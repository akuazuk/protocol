#!/usr/bin/env python3
"""Проверить daily MIS Parquet blocking/warning gates."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import atomic_write_json, validate_export


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--source-rows", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate_export(
        pd.read_parquet(args.parquet),
        day=args.date,
        source_rows=args.source_rows,
    )
    payload = result.to_dict()
    if args.out:
        atomic_write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
