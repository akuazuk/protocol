#!/usr/bin/env python3
"""Собрать компактные паспорта КП из локальной выгрузки Ilex (без HTML в git)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.ilex_protocol_passports import (  # noqa: E402
    DEFAULT_PASSPORTS,
    build_passports_from_dump,
    write_passports_jsonl,
)

DEFAULT_DUMP = Path.home() / "Protocol_Private" / "ilex_clinical_protocols"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--out", type=Path, default=DEFAULT_PASSPORTS)
    args = parser.parse_args()
    rows = build_passports_from_dump(args.dump)
    write_passports_jsonl(rows, args.out)
    status = Counter(str(r.get("status") or "?") for r in rows)
    with_icd = sum(1 for r in rows if r.get("icd10_primary"))
    unique_titles = len({str(r.get("protocol_title") or "").casefold() for r in rows})
    summary = {
        "out": str(args.out),
        "rows": len(rows),
        "unique_titles": unique_titles,
        "with_icd": with_icd,
        "status": dict(status),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
