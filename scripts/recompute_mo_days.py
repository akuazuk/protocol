#!/usr/bin/env python3
"""Пересобрать отчёты и витрину МО из уже сохранённых артефактов дня.

Без обращения к МИС и без повторной оценки: берём сохранённую партицию
`raw/YYYY/MM/mo_YYYY-MM-DD.parquet` и оценки
`secure_cases/YYYY/MM/kz_l1_YYYY-MM-DD_cases.jsonl`. Нужен, когда меняется
методика агрегации (шкала балла, оси, покрытие) и историю надо привести к ней
же, не тратя выгрузку и LLM.
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

from clinical_knowledge.mo_daily import (  # noqa: E402
    add_document_taxonomy,
    assess_completeness,
    build_daily_report,
    case_overall_pct,
    initialize_warehouse,
    load_jsonl,
    upsert_warehouse,
    write_daily_report,
)


def _days(first: date, last: date) -> list[date]:
    span = (last - first).days
    return [first + timedelta(days=offset) for offset in range(span + 1)]


def _llm_queue_pending(secure_dir: Path, day: date) -> int:
    path = secure_dir / f"kz_l1_{day.isoformat()}_llm_queue.json"
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    items = payload.get("pending") if isinstance(payload, dict) else payload
    return len(items) if isinstance(items, list) else 0


def recompute_day(
    day: date,
    *,
    data_root: Path,
    warehouse: Path,
    write_reports: bool = True,
) -> dict[str, Any]:
    import pandas as pd

    partition = data_root / "raw" / f"{day:%Y}" / f"{day:%m}" / f"mo_{day.isoformat()}.parquet"
    secure_dir = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    cases_path = secure_dir / f"kz_l1_{day.isoformat()}_cases.jsonl"
    if not partition.is_file():
        return {"date": day.isoformat(), "status": "missing_partition"}
    frame = add_document_taxonomy(pd.read_parquet(partition))
    raw_rows = frame.drop(columns=["result_raw"], errors="ignore").to_dict(orient="records")
    cases = load_jsonl(cases_path)
    report_path = data_root / "reports" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / "report.json"
    previous: dict[str, Any] = {}
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    completeness = assess_completeness(
        raw_rows, cases, llm_queue_pending=_llm_queue_pending(secure_dir, day)
    )
    report, public = build_daily_report(
        raw_rows,
        cases,
        day=day,
        run_id=str(previous.get("run_id") or f"recompute-{day.isoformat()}"),
        revision=int(previous.get("revision") or 1),
        quality=dict(previous.get("quality") or {"passed": True, "recomputed": True}),
        month_to_date=previous.get("month_to_date"),
        comparisons=previous.get("comparisons"),
        completeness=completeness,
    )
    if write_reports:
        write_daily_report(report, public, day=day, root=data_root)
    upsert_warehouse(warehouse, raw_rows, cases, report)
    scores = [score for row in cases if (score := case_overall_pct(row)) is not None]
    return {
        "date": day.isoformat(),
        "status": "success",
        "rows": len(raw_rows),
        "cases": len(cases),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "coverage_pct": completeness["coverage_pct"],
        "partial": completeness["partial"],
        "was_avg_score": (previous.get("summary") or {}).get("avg_score"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--last-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=1),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "medical_exams",
        help="каталог MO_DATA_ROOT с raw/secure_cases/reports",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=None,
        help="файл витрины; по умолчанию <data-root>/warehouse/mo_analytics.sqlite",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="обновить только витрину, не перезаписывая report.json и публичные снапшоты",
    )
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser()
    warehouse = (args.warehouse or data_root / "warehouse" / "mo_analytics.sqlite").expanduser()
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    initialize_warehouse(warehouse)
    results = [
        recompute_day(
            day,
            data_root=data_root,
            warehouse=warehouse,
            write_reports=not args.skip_reports,
        )
        for day in _days(args.first_date, args.last_date)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"success", "missing_partition"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
