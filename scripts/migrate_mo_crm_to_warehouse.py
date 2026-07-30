#!/usr/bin/env python3
"""Перенос операционных таблиц CRM из старого файла в единую витрину МО.

Источник открывается только на чтение и остаётся на диске как резервная копия.
Повторный запуск безопасен: строки с существующими ключами не дублируются.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import CRM_TABLES, initialize_warehouse, migrate_crm


def _counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        available = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in CRM_TABLES:
            if table in available:
                out[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "ml" / "secure" / "mo_methodist.sqlite",
        help="старый файл CRM",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "data" / "medical_exams" / "warehouse" / "mo_analytics.sqlite",
        help="единая витрина",
    )
    parser.add_argument("--dry-run", action="store_true", help="только показать, что будет перенесено")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.expanduser()
    target = args.target.expanduser()
    report = {
        "source": str(source),
        "target": str(target),
        "source_rows": _counts(source),
        "target_rows_before": _counts(target),
    }
    if not source.is_file():
        report["status"] = "source_missing"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        report["status"] = "dry_run"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    initialize_warehouse(target)
    report["moved"] = migrate_crm(source, target)
    report["target_rows_after"] = _counts(target)
    report["status"] = "migrated"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
