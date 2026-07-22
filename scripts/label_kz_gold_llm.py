#!/usr/bin/env python3
"""Э4.2: LLM-предразметка gold-выборки (proxy-метки для калибровки осей).

Для каждого visit_id из gold-манифеста вызывает полный L2+LLM-обзор
(`review_one_visit_full`) и пишет компактную метку {visit_id, llm_overall_pct,
llm_status} в labels jsonl. Резюмируемо: уже размеченные visit_id пропускаются.

Это НЕ замена методисту (ручной gold - следующая итерация), а воспроизводимая
proxy-метка от сильной модели для калибровки порогов axes-overall.

  PYTHONPATH=. python3 scripts/label_kz_gold_llm.py \\
    --gold /var/data/mis_protocol/kz_gold/gold_sample.jsonl \\
    --out  /var/data/mis_protocol/kz_gold/gold_llm_labels.jsonl \\
    --month 2026-07 --limit 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(r.get("visit_id") or "")
            if vid and not r.get("error"):
                done.add(vid)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--limit", type=int, default=0, help="0 = все")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    from clinical_knowledge.mis_kz_quality import review_one_visit_full

    gold: list[dict] = []
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                gold.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    done = _load_done(args.out)
    todo = [g for g in gold if str(g.get("visit_id") or "") not in done]
    if args.limit > 0:
        todo = todo[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"gold={len(gold)} done={len(done)} todo={len(todo)}", flush=True)

    ok = fail = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, g in enumerate(todo, 1):
            vid = str(g.get("visit_id") or "")
            try:
                res = review_one_visit_full(month=args.month, visit_id=vid)
                item = res.get("item") or {}
                rec = {
                    "visit_id": vid,
                    "specialty": g.get("specialty"),
                    "band": g.get("band"),
                    "l1_overall_pct": g.get("overall_pct"),
                    "llm_overall_pct": item.get("overall_pct"),
                    "llm_status": item.get("status"),
                    "n_critical_gaps": len(item.get("critical_gaps_ru") or []),
                    "error": None if res.get("ok") else (res.get("error") or "unknown"),
                }
                if res.get("ok"):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:  # noqa: BLE001
                rec = {"visit_id": vid, "error": str(e)[:200]}
                fail += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"DONE ok={ok} fail={fail} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
