#!/usr/bin/env python3
"""Собрать паспорта клиентов со склада МО (волна P0).

Без PHI в stdout: только счётчики. Оценку визитов не запускает.

  PYTHONPATH=. python3 scripts/build_mo_patient_passports.py
  PYTHONPATH=. python3 scripts/build_mo_patient_passports.py --keys-file /tmp/keys.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_patient_passport import (  # noqa: E402
    default_lab_path,
    default_warehouse_path,
    rebuild_passports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild patient passports from warehouse")
    parser.add_argument("--warehouse", type=Path, default=None)
    parser.add_argument("--lab", type=Path, default=None)
    parser.add_argument("--keys-file", type=Path, default=None, help="patient_key per line")
    args = parser.parse_args()
    warehouse = args.warehouse or default_warehouse_path()
    if warehouse is None or not Path(warehouse).is_file():
        print(json.dumps({"ok": False, "error": "warehouse_missing"}, ensure_ascii=False))
        return 2
    keys = None
    if args.keys_file:
        keys = [
            line.strip()
            for line in args.keys_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    report = rebuild_passports(
        Path(warehouse),
        lab_path=args.lab or default_lab_path(Path(warehouse)),
        patient_keys=keys,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
