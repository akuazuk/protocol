#!/usr/bin/env python3
"""Утренний Telegram-брифинг МО за вчера (фаза 8.2).

Пример:
  python3 scripts/ops/mo_morning_briefing.py
  python3 scripts/ops/mo_morning_briefing.py --date 2026-08-02 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MINSK = ZoneInfo("Europe/Minsk")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send MO morning Telegram briefing")
    parser.add_argument("--date", default="", help="YYYY-MM-DD, default = yesterday Minsk")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--base-url",
        default="https://protocol-bimy.onrender.com",
        help="Public app URL for slice links",
    )
    args = parser.parse_args()
    day = args.date.strip() or (datetime.now(MINSK).date() - timedelta(days=1)).isoformat()

    from clinical_knowledge.mo_backend import build_daily_report
    from clinical_knowledge.mo_report_engine import build_telegram_briefing
    from scripts.telegram_notify import send_telegram

    report = build_daily_report(day)
    text = build_telegram_briefing(report, base_url=args.base_url)
    print(text)
    if args.dry_run:
        return 0
    ok = send_telegram(text)
    if not ok:
        print("Telegram notify skipped or failed (check TELEGRAM_* env)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
