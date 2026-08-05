#!/usr/bin/env python3
"""Пересобрать fact_/dim_ витрины из secure CSV + cases.jsonl (без МИС и без LLM).

Нужен, когда publish сделал `INSERT SELECT *` в таблицу с другим порядком колонок
и doctor_key/specialty/filial оказались сдвинуты.

Пример на Render:
  python3 scripts/repair_mo_warehouse_from_secure.py \\
    --data-root /var/data/medical_exams \\
    --first-date 2026-08-01 --last-date 2026-08-04 \\
    --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import (  # noqa: E402
    assess_completeness,
    build_daily_report,
    initialize_warehouse,
    load_jsonl,
    upsert_warehouse,
    write_daily_report,
)
from clinical_knowledge.mo_publish import common_columns, merge_sql  # noqa: E402


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _llm_queue_pending(secure_dir: Path, day: date) -> int:
    path = secure_dir / f"kz_l1_{day.isoformat()}_llm_queue.json"
    graded_path = secure_dir / f"kz_l1_{day.isoformat()}_llm_grades.jsonl"
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    queued: set[str] = set()
    if isinstance(payload, dict):
        for key in ("visit_ids", "pending", "queue", "items", "cases"):
            value = payload.get(key)
            if isinstance(value, list):
                queued = {
                    str(item.get("visit_id") if isinstance(item, dict) else item)
                    for item in value
                    if item not in (None, "")
                }
                break
    graded: set[str] = set()
    if graded_path.is_file():
        for row in load_jsonl(graded_path):
            if row.get("_error") or row.get("error"):
                continue
            vid = str(row.get("visit_id") or row.get("case_id") or "")
            if vid:
                graded.add(vid)
    return len(queued - graded) if queued else 0


def _load_raw_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not item.get("document_kind"):
            item["document_kind"] = item.get("kz_kind") or item.get("doc_type") or "unknown"
        if not item.get("visit_date"):
            item["visit_date"] = (item.get("date") or item.get("visit_date_iso_db") or "")[:10]
        out.append(item)
    return out


def rebuild_day(
    day: date,
    *,
    data_root: Path,
    warehouse: Path,
    write_reports: bool,
) -> dict[str, Any]:
    secure_dir = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    csv_path = secure_dir / f"mo_{day.isoformat()}.csv"
    cases_path = secure_dir / f"kz_l1_{day.isoformat()}_cases.jsonl"
    if not csv_path.is_file() or not cases_path.is_file():
        return {
            "date": day.isoformat(),
            "status": "missing_artifacts",
            "csv": csv_path.is_file(),
            "cases": cases_path.is_file(),
        }
    raw_rows = _load_raw_csv(csv_path)
    cases = load_jsonl(cases_path)
    completeness = assess_completeness(
        raw_rows, cases, llm_queue_pending=_llm_queue_pending(secure_dir, day)
    )
    report_path = data_root / "reports" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / "report.json"
    previous: dict[str, Any] = {}
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    report, public = build_daily_report(
        raw_rows,
        cases,
        day=day,
        run_id=str(previous.get("run_id") or f"repair-{day.isoformat()}"),
        revision=int(previous.get("revision") or 1) + (1 if write_reports else 0),
        quality=dict(previous.get("quality") or {"passed": True, "repaired": True}),
        month_to_date=previous.get("month_to_date"),
        comparisons=previous.get("comparisons"),
        completeness=completeness,
    )
    if write_reports:
        write_daily_report(report, public, day=day, root=data_root)
    written = upsert_warehouse(warehouse, raw_rows, cases, report)
    return {
        "date": day.isoformat(),
        "status": "success",
        "rows": len(raw_rows),
        "cases": len(cases),
        "llm_queue_pending": completeness.get("llm_queue_pending"),
        "partial": completeness.get("partial"),
        "written": dict(written),
    }


def merge_into_prod(repair_db: Path, prod_db: Path) -> dict[str, int]:
    initialize_warehouse(prod_db)
    counts: dict[str, int] = {}
    with sqlite3.connect(prod_db) as prod, sqlite3.connect(repair_db) as repair:
        tables = [
            row[0]
            for row in repair.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if row[0].startswith(("fact_", "dim_"))
        ]
        column_map = {
            table: common_columns(prod, repair, table)
            for table in tables
            if common_columns(prod, repair, table)
        }
        # удаляем из прода дни, которые есть в repair, затем доливаем named-columns
        days = [
            row[0]
            for row in repair.execute(
                "SELECT DISTINCT visit_date FROM fact_mo_case ORDER BY 1"
            )
        ]
        if days:
            placeholders = ",".join("?" for _ in days)
            for table, date_col in (
                ("fact_mo_case", "visit_date"),
                ("fact_mo_doctor_daily", "visit_date"),
                ("fact_mo_daily", "visit_date"),
                ("fact_mo_visit", "visit_date"),
            ):
                exists = prod.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if exists:
                    prod.execute(
                        f"DELETE FROM {table} WHERE {date_col} IN ({placeholders})",
                        days,
                    )
            # findings/axes for deleted cases
            mis_ids = [
                row[0]
                for row in repair.execute("SELECT mis_id FROM fact_mo_case")
            ]
            if mis_ids:
                chunk = 500
                for start in range(0, len(mis_ids), chunk):
                    part = mis_ids[start : start + chunk]
                    ph = ",".join("?" for _ in part)
                    for table in ("fact_mo_finding", "fact_mo_score_axis"):
                        if prod.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (table,),
                        ).fetchone():
                            prod.execute(f"DELETE FROM {table} WHERE mis_id IN ({ph})", part)
        prod.commit()

    sql = merge_sql(tables, snapshot_path=str(repair_db.resolve()), column_map=column_map)
    with sqlite3.connect(prod_db) as prod:
        prod.executescript(sql)
        for table in tables:
            counts[table] = prod.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--first-date", type=date.fromisoformat, required=True)
    ap.add_argument("--last-date", type=date.fromisoformat, required=True)
    ap.add_argument(
        "--warehouse",
        type=Path,
        default=None,
        help="продовая витрина; по умолчанию <data-root>/warehouse/mo_analytics.sqlite",
    )
    ap.add_argument("--skip-reports", action="store_true")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="влить repair-витрину в продовую; без флага только dry rebuild в tmp",
    )
    args = ap.parse_args(argv)
    data_root = args.data_root.expanduser()
    prod = (args.warehouse or data_root / "warehouse" / "mo_analytics.sqlite").expanduser()
    with tempfile.TemporaryDirectory(prefix="mo-repair-") as tmp:
        repair = Path(tmp) / "repair.sqlite"
        initialize_warehouse(repair)
        results = [
            rebuild_day(
                day,
                data_root=data_root,
                warehouse=repair,
                write_reports=args.apply and not args.skip_reports,
            )
            for day in _days(args.first_date, args.last_date)
        ]
        print(json.dumps({"rebuild": results}, ensure_ascii=False, indent=2))
        sample = None
        with sqlite3.connect(repair) as db:
            sample = db.execute(
                "SELECT doctor_key, specialty, filial, status, overall_pct "
                "FROM fact_mo_case WHERE doctor_key != '' LIMIT 3"
            ).fetchall()
        print(json.dumps({"repair_sample": sample}, ensure_ascii=False))
        if not args.apply:
            print("dry-run: укажите --apply чтобы влить в продовую витрину")
            return 0 if all(r["status"] in {"success", "missing_artifacts"} for r in results) else 1
        counts = merge_into_prod(repair, prod)
        print(json.dumps({"merged": counts, "prod": str(prod)}, ensure_ascii=False, indent=2))
        with sqlite3.connect(prod) as db:
            top = db.execute(
                "SELECT doctor_key, specialty, filial, count(*) n FROM fact_mo_case "
                "WHERE visit_date >= ? GROUP BY 1,2,3 ORDER BY n DESC LIMIT 8",
                (args.first_date.isoformat(),),
            ).fetchall()
        print(json.dumps({"prod_top_after": top}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
