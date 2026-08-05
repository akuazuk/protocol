#!/usr/bin/env python3
"""Переклассифицировать дни, где partial держался только из-за llm_queue_pending.

Применяет текущую политику completeness к report.json / public.json, pipeline.json
и quality_status в fact_mo_daily. Не пересчитывает оценки.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import apply_completeness_policy, atomic_write_json  # noqa: E402


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _report_paths(data_root: Path, day: date) -> tuple[Path, Path]:
    folder = data_root / "reports" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    return folder / "report.json", folder / "public.json"


def reclassify_day(data_root: Path, day: date, *, dry_run: bool = False) -> dict[str, Any]:
    report_path, public_path = _report_paths(data_root, day)
    report = _load_json(report_path)
    if report is None:
        return {"date": day.isoformat(), "status": "missing_report"}
    before = dict(report.get("completeness") or {})
    after = apply_completeness_policy(
        {
            **before,
            "llm_queue_pending": before.get("llm_queue_pending")
            or (report.get("completeness") or {}).get("llm_queue_pending")
            or 0,
            "partial": report.get("partial"),
        }
    )
    changed = bool(report.get("partial")) != bool(after["partial"]) or before.get("reasons") != after.get(
        "reasons"
    )
    result: dict[str, Any] = {
        "date": day.isoformat(),
        "changed": changed,
        "before_partial": bool(report.get("partial")),
        "after_partial": bool(after["partial"]),
        "before_reasons": before.get("reasons") or [],
        "after_reasons": after.get("reasons") or [],
        "advisory_reasons": after.get("advisory_reasons") or [],
        "dry_run": dry_run,
    }
    if not changed or dry_run:
        return result

    report["completeness"] = after
    report["partial"] = after["partial"]
    atomic_write_json(report_path, report)

    public = _load_json(public_path)
    if public is not None:
        public["completeness"] = after
        public["partial"] = after["partial"]
        atomic_write_json(public_path, public)

    state_path = data_root / "state" / "pipeline.json"
    state = _load_json(state_path)
    if state is not None:
        dates = state.setdefault("dates", {})
        entry = dates.setdefault(day.isoformat(), {})
        entry["completeness"] = after
        if after["partial"]:
            entry["status"] = "partial"
        elif entry.get("status") == "partial":
            entry["status"] = "success"
        atomic_write_json(state_path, state)

    warehouse = data_root / "warehouse" / "mo_analytics.sqlite"
    if warehouse.is_file():
        quality = "partial" if after["partial"] else "passed"
        with sqlite3.connect(warehouse) as db:
            db.execute(
                "UPDATE fact_mo_daily SET partial = ?, quality_status = ? WHERE visit_date = ?",
                (1 if after["partial"] else 0, quality, day.isoformat()),
            )
            db.commit()
        result["warehouse_quality_status"] = quality
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "medical_exams",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="даты YYYY-MM-DD",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    results = [
        reclassify_day(args.data_root.expanduser(), date.fromisoformat(item), dry_run=args.dry_run)
        for item in args.dates
    ]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
