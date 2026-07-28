#!/usr/bin/env python3
"""Единый entrypoint ежедневного МО-pipeline и генератора отчёта."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_orchestrator import MoDailyPipeline, PipelinePaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="yesterday", help="yesterday или YYYY-MM-DD")
    parser.add_argument("--catch-up", action="store_true", help="обработать пропущенные даты")
    parser.add_argument("--catch-up-limit", type=int, default=31)
    parser.add_argument("--first-date", type=date.fromisoformat)
    parser.add_argument("--reconcile-days", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="показать команды без файлов, SQL и VPN")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "medical_exams",
        help="локальный защищённый root",
    )
    parser.add_argument("--no-telegram", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    notify = None
    if not args.no_telegram and not args.dry_run:
        from scripts.telegram_notify import send_telegram

        notify = send_telegram
    pipeline = MoDailyPipeline(
        PipelinePaths(ROOT, args.data_root.expanduser().resolve()),
        dry_run=args.dry_run,
        notify=notify,
    )
    results = pipeline.run(
        date_value=args.date,
        catch_up=args.catch_up,
        catch_up_limit=max(1, args.catch_up_limit),
        first_date=args.first_date,
        reconcile_days=max(0, args.reconcile_days),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
