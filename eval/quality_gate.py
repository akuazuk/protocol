#!/usr/bin/env python3
"""Порог качества поиска: читает отчёт search_quality_eval и проваливается, если pass_rate ниже порога.

Зачем отдельно от exit-кода eval: одиночные «пограничные» golden-кейсы не должны блокировать
релиз, но просадка ниже порога (регрессия отбора) должна.

Использование:

  # 1) Сформировать отчёт (нужен корпус; для прод-метрик — ключ API и --embed-on)
  python3 eval/search_quality_eval.py --golden eval/golden_queries.jsonl --report-json report.json

  # 2) Проверить порог (по умолчанию 0.9 или QUALITY_MIN_PASS_RATE)
  python3 eval/quality_gate.py --report report.json --min-pass-rate 0.9

Коды возврата: 0 — порог пройден, 1 — ниже порога, 2 — ошибка чтения отчёта.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_summary(report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if isinstance(summary, dict) and "pass_rate" in summary:
        return summary
    cases = data.get("cases") or []
    total = len(cases)
    passed = sum(1 for c in cases if c.get("ok"))
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Порог качества по отчёту search_quality_eval")
    ap.add_argument("--report", type=Path, required=True, help="JSON-отчёт (--report-json)")
    ap.add_argument(
        "--min-pass-rate",
        type=float,
        default=float(os.environ.get("QUALITY_MIN_PASS_RATE", "0.9")),
        help="минимальная доля пройденных кейсов (0..1), по умолчанию 0.9 / QUALITY_MIN_PASS_RATE",
    )
    args = ap.parse_args()

    if not args.report.is_file():
        print(f"Отчёт не найден: {args.report}", file=sys.stderr)
        return 2
    try:
        summary = _load_summary(args.report)
    except (OSError, ValueError) as e:
        print(f"Не удалось прочитать отчёт: {e}", file=sys.stderr)
        return 2

    rate = float(summary.get("pass_rate") or 0.0)
    total = int(summary.get("total") or 0)
    passed = int(summary.get("passed") or 0)
    print(
        f"Качество поиска: {passed}/{total} кейсов пройдено (pass_rate={rate:.2%}), "
        f"порог={args.min_pass_rate:.2%}"
    )
    if total == 0:
        print("Нет кейсов в отчёте — порог не может быть проверён.", file=sys.stderr)
        return 2
    if rate + 1e-9 < args.min_pass_rate:
        print("НИЖЕ ПОРОГА: возможна регрессия отбора протоколов.", file=sys.stderr)
        return 1
    print("Порог пройден.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
