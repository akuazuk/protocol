#!/usr/bin/env python3
"""Пересчитать deep/findings в kz_l1_*_cases.jsonl новым кодом (без МИС).

Пример на Render:
  .venv/bin/python scripts/rescore_mo_deep_days.py \\
    --data-root /var/data/medical_exams \\
    --first-date 2026-08-01 --last-date 2026-08-05
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep, load_drug_ctx  # noqa: E402


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _case_for_deep(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    clinical = row.get("clinical") if isinstance(row.get("clinical"), dict) else {}
    for key, value in clinical.items():
        if value and not out.get(key):
            out[key] = value
    return out


def rescore_day(day: date, *, data_root: Path) -> dict[str, Any]:
    secure = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    path = secure / f"kz_l1_{day.isoformat()}_cases.jsonl"
    if not path.is_file():
        return {"date": day.isoformat(), "status": "missing_cases"}
    drug_ctx = load_drug_ctx()
    rows: list[dict[str, Any]] = []
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        before = json.dumps(row.get("deep") or {}, ensure_ascii=False, sort_keys=True)
        deep = evaluate_kz_deep(_case_for_deep(row), protocol_ctx=None, drug_ctx=drug_ctx)
        row["deep"] = deep
        if deep.get("overall_pct") is not None:
            row["overall_pct"] = deep.get("overall_pct")
        after = json.dumps(row.get("deep") or {}, ensure_ascii=False, sort_keys=True)
        if before != after:
            changed += 1
        rows.append(row)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return {
        "date": day.isoformat(),
        "status": "success",
        "cases": len(rows),
        "changed": changed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--first-date", type=date.fromisoformat, required=True)
    ap.add_argument("--last-date", type=date.fromisoformat, required=True)
    args = ap.parse_args(argv)
    results = [
        rescore_day(day, data_root=args.data_root.expanduser())
        for day in _days(args.first_date, args.last_date)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"success", "missing_cases"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
