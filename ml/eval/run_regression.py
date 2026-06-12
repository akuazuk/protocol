#!/usr/bin/env python3
"""Регрессия KZ и search quality для ML-релизов."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ml" / "configs" / "default.json"
GOLD = ROOT / "data" / "gastro_mvp" / "consult_gold.jsonl"
BENCH = ROOT / "data" / "quality_benchmark.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else {}
    eval_cfg = cfg.get("eval") or {}

    kz_n = sum(1 for _ in GOLD.open(encoding="utf-8") if _.strip()) if GOLD.is_file() else 0
    bench_n = 0
    if BENCH.is_file():
        data = json.loads(BENCH.read_text(encoding="utf-8"))
        bench_n = len(data.get("queries") or [])

    report = {
        "kz_gold_cases": kz_n,
        "search_golden_queries": bench_n,
        "min_kz_regression_pass_rate": eval_cfg.get("min_kz_regression_pass_rate", 0.85),
        "min_search_pass_rate": eval_cfg.get("min_search_pass_rate", 0.89),
        "status": "stub",
        "message": "Wire pytest/eval/search_quality_eval.py before promoting model in registry",
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"KZ gold cases: {kz_n}")
        print(f"Search golden: {bench_n}")
        print("Status: stub - run full eval before model promotion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
