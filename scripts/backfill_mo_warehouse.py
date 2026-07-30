#!/usr/bin/env python3
"""Перенести уже выгруженную историю 2026 года в локальную витрину МО."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import (  # noqa: E402
    add_document_taxonomy,
    build_daily_report,
    initialize_warehouse,
    upsert_warehouse,
)


def _months(first: str, last: str) -> list[str]:
    current = date.fromisoformat(first + "-01")
    stop = date.fromisoformat(last + "-01")
    result: list[str] = []
    while current <= stop:
        result.append(current.strftime("%Y-%m"))
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


def _load_cases(month: str, data_root: Path) -> list[dict[str, Any]]:
    candidates = [
        data_root / "ml" / "reports" / "deep_eval" / f"kz_l1_{month}_cases.jsonl",
        data_root / "mis_protocol" / f"kz_l1_{month}_cases.jsonl",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return []
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def backfill_month(
    month: str,
    *,
    warehouse: Path,
    through_date: date,
    dry_run: bool = False,
    data_root: Path | None = None,
) -> dict[str, Any]:
    data_root = data_root or ROOT / "data"
    csv_path = data_root / "mis_protocol" / f"mis_protocol_{month}.csv"
    if not csv_path.is_file():
        return {"month": month, "status": "missing_csv", "rows": 0}
    frame = add_document_taxonomy(pd.read_csv(csv_path, low_memory=False))
    frame["visit_date"] = frame["visit_date"].fillna(frame.get("date", "")).astype(str).str[:10]
    frame = frame[frame["visit_date"] <= through_date.isoformat()].copy()
    raw_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        if len(row["visit_date"]) == 10:
            raw_by_day[row["visit_date"]].append(row)
    cases = _load_cases(month, data_root)
    cases_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        day = str(case.get("date") or "")[:10]
        if len(day) == 10:
            cases_by_day[day].append(case)
    if not dry_run:
        for day, raw_rows in sorted(raw_by_day.items()):
            chosen = date.fromisoformat(day)
            day_cases = cases_by_day.get(day, [])
            secure_report, _ = build_daily_report(
                raw_rows,
                day_cases,
                day=chosen,
                run_id=f"history-{month}",
                revision=1,
                quality={"passed": True, "backfill": True},
            )
            upsert_warehouse(warehouse, raw_rows, day_cases, secure_report)
    return {
        "month": month,
        "status": "dry_run" if dry_run else "success",
        "rows": len(frame),
        "cases": len(cases),
        "days": len(raw_by_day),
    }


def prune_after(warehouse: Path, through_date: date) -> int:
    if not warehouse.is_file():
        return 0
    with sqlite3.connect(warehouse) as db:
        stale_ids = [
            row[0]
            for row in db.execute(
                "SELECT mis_id FROM fact_mo_case WHERE visit_date > ?",
                (through_date.isoformat(),),
            )
        ]
        if stale_ids:
            marks = ",".join("?" for _ in stale_ids)
            db.execute(f"DELETE FROM fact_mo_finding WHERE mis_id IN ({marks})", stale_ids)
            db.execute(f"DELETE FROM fact_mo_score_axis WHERE mis_id IN ({marks})", stale_ids)
            db.execute(f"DELETE FROM fact_mo_case WHERE mis_id IN ({marks})", stale_ids)
        db.execute("DELETE FROM fact_mo_daily WHERE visit_date > ?", (through_date.isoformat(),))
        db.commit()
    return len(stale_ids)


def propagate_visit_scores(warehouse: Path) -> int:
    """Заполнить дубли документа оценкой того же визита.

    Scoring сводит несколько КЗ к одному случаю на `visit_id`. Записи одного
    визита могут иметь разные `mis_id` и даже даты, поэтому дневной upsert не
    всегда видит выбранный scorer-ом случай. Переносим балл только если внутри
    визита существует ровно одно значение оценки - неоднозначные визиты не
    трогаем.
    """
    with sqlite3.connect(warehouse) as db:
        before = db.total_changes
        db.execute(
            """UPDATE fact_mo_case AS target
               SET overall_pct = (
                       SELECT MIN(source.overall_pct)
                       FROM fact_mo_case AS source
                       WHERE source.visit_id = target.visit_id
                         AND source.overall_pct IS NOT NULL
                       GROUP BY source.visit_id
                       HAVING COUNT(DISTINCT source.overall_pct) = 1
                   ),
                   status = COALESCE(NULLIF(target.status, ''), (
                       SELECT MIN(NULLIF(source.status, ''))
                       FROM fact_mo_case AS source
                       WHERE source.visit_id = target.visit_id
                         AND source.overall_pct IS NOT NULL
                   ))
               WHERE target.overall_pct IS NULL
                 AND target.document_kind IN ('medical_exam', 'consultation')
                 AND EXISTS (
                       SELECT 1
                       FROM fact_mo_case AS source
                       WHERE source.visit_id = target.visit_id
                         AND source.overall_pct IS NOT NULL
                       GROUP BY source.visit_id
                       HAVING COUNT(DISTINCT source.overall_pct) = 1
                 )"""
        )
        changed = db.total_changes - before
        db.commit()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-month", default="2026-01")
    parser.add_argument("--last-month", default=date.today().strftime("%Y-%m"))
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=ROOT / "data" / "medical_exams" / "warehouse" / "mo_analytics.sqlite",
    )
    parser.add_argument(
        "--through-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=1),
        help="последняя дата, включаемая в витрину; по умолчанию вчера",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="каталог data/ с выгрузками МИС и оценками (по умолчанию data/ этого чекаута)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="не удалять дни после --through-date: их ведёт ежедневный конвейер",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    warehouse = args.warehouse.expanduser()
    if not args.dry_run:
        warehouse.parent.mkdir(parents=True, exist_ok=True)
        initialize_warehouse(warehouse)
    results = [
        backfill_month(
            month,
            warehouse=warehouse,
            through_date=args.through_date,
            dry_run=args.dry_run,
            data_root=args.data_root.expanduser(),
        )
        for month in _months(args.first_month, args.last_month)
    ]
    if not args.dry_run and not args.no_prune:
        pruned = prune_after(warehouse, args.through_date)
        if pruned:
            print(f"Удалено строк после {args.through_date.isoformat()}: {pruned}", file=sys.stderr)
    if not args.dry_run:
        propagated = propagate_visit_scores(warehouse)
        if propagated:
            print(f"Заполнено дублей документов оценкой визита: {propagated}", file=sys.stderr)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"success", "dry_run"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
