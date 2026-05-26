#!/usr/bin/env python3
"""Обновить data/quality_benchmark.json по eval/golden_queries.prod.jsonl (лексический retrieve).

Запуск из корня репозитория:
  python3 scripts/update_quality_benchmark.py
  python3 scripts/update_quality_benchmark.py --mini
  python3 scripts/update_quality_benchmark.py --golden eval/golden_queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true", help="мини-корпус tests/fixtures")
    ap.add_argument("--golden", default="eval/golden_queries.prod.jsonl")
    ap.add_argument("--out", default="data/quality_benchmark.json")
    args = ap.parse_args()

    if args.mini:
        os.environ["RAG_CHUNKS_JSONL"] = str(ROOT / "tests/fixtures/chunks.mini.jsonl")
        golden = ROOT / "eval/golden_queries.jsonl"
    else:
        golden = ROOT / args.golden
        if not golden.is_file():
            golden = ROOT / "eval/golden_queries.prod.example.jsonl"

    from eval.query_tester import load_golden_lines  # noqa: E402
    from eval.search_quality_eval import evaluate_one  # noqa: E402
    from rag_server import load_data, retrieve  # noqa: E402

    load_data()
    cases = load_golden_lines(golden)
    passed = 0
    failed_labels: list[str] = []
    for j, case in enumerate(cases, 1):
        q = (case.get("query") or "").strip()
        if not q:
            continue
        rep = evaluate_one(j, case, retrieve, max_chunks=6, max_per_path=2, gemini_advice=False, api_key_present=False, embed_requested=False)
        if rep.ok:
            passed += 1
        else:
            failed_labels.append(q[:60])

    total = len([c for c in cases if (c.get("query") or "").strip()])
    pct = round(100.0 * passed / total) if total else 0
    out_path = ROOT / args.out
    payload = {
        "title": "Эталонная проверка подбора протоколов",
        "corpus": "мини-корпус" if args.mini else "полный корпус RAG_CHUNKS_*",
        "queries_total": total,
        "queries_passed": passed,
        "pass_rate_pct": pct,
        "failed_queries_sample": failed_labels[:8],
        "methodology_ru": f"golden: {golden.name}; retrieve() + eval/search_quality_eval.evaluate_one",
        "prod_note_ru": "Перезапуск: python3 scripts/update_quality_benchmark.py",
        "updated": date.today().isoformat(),
        "metrics": [
            {"label": "Эталонных запросов", "value": str(total), "hint": golden.name},
            {"label": "Успешных проверок", "value": f"{passed}/{total}", "hint": "must_substrings / expect_empty"},
            {"label": "Рубрики каталога Минздрава", "value": "24", "hint": "verify_minzdrav_rubrics.py"},
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} — {passed}/{total} ({pct}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
