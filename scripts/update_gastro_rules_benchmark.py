#!/usr/bin/env python3
"""Пересчитать data/gastro_mvp/benchmark.json по consult_gold.jsonl."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.benchmark import write_gastro_benchmark


def main() -> int:
    payload = write_gastro_benchmark()
    total = payload.get("cases_total", 0)
    passed = payload.get("cases_passed", 0)
    print(f"Wrote data/gastro_mvp/benchmark.json — {passed}/{total} ({payload.get('pass_rate_pct')}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
