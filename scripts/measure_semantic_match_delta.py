#!/usr/bin/env python3
"""Э2 верификация на реальных КЗ: A/B дельта exams/treatment от каталога синонимов.

Один процесс, одна выборка КЗ, два прогона L1: каталог ВЫКЛ (базлайн, как было) и ВКЛ.
Чистое сравнение эффекта семантического матча без влияния другой выборки.

Запуск на Render (данные на /var/data, тёплый код после деплоя):

  PYTHONPATH=. python3 scripts/measure_semantic_match_delta.py \\
    --csv /var/data/mis_protocol/mis_protocol_2026-07.csv --limit 500

Печатает средние exams/treatment для OFF vs ON и дельту.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import os
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "clinical_knowledge").is_dir():
    ROOT = Path(os.environ.get("PROTOCOL_ROOT") or "/opt/render/project/src")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mis_protocol_l1_batch import (  # noqa: E402
    _direct_tier,
    build_kz_text,
    select_rows_for_l1,
)


def _block_scores(result: dict) -> dict[str, float]:
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") if isinstance(sa, dict) else {}
    comp = comp if isinstance(comp, dict) else {}
    alignment = comp.get("alignment_by_block") or {}
    out: dict[str, float] = {}
    if isinstance(alignment, dict):
        for bid, val in alignment.items():
            sc = None
            if isinstance(val, dict):
                sc = val.get("score", val.get("alignment_score"))
            elif isinstance(val, (int, float)):
                sc = float(val)
            if isinstance(sc, (int, float)):
                out[str(bid)] = float(sc)
    return out


def _run(rows: list[dict], tag: str) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {"exams": [], "treatment": [], "overall": []}
    for i, row in enumerate(rows, 1):
        vid = str(row.get("visit_id") or row.get("id") or "")
        try:
            # префикс "mis-" обязателен: id на "ab" спец-обрабатывается движком → not_assessed;
            # разные tag для OFF/ON, чтобы исключить кэш по consultation_id.
            res = _direct_tier(build_kz_text(row), f"mis-{tag}-{vid}")
        except Exception:
            continue
        bs = _block_scores(res)
        if "exams" in bs:
            acc["exams"].append(bs["exams"])
        if "treatment" in bs:
            acc["treatment"].append(bs["treatment"])
        ov = res.get("overall_score")
        if isinstance(ov, (int, float)):
            acc["overall"].append(float(ov))
        if i % 100 == 0:
            print(f"  ... {i}/{len(rows)}", flush=True)
    return acc


def _fmt(acc: dict[str, list[float]]) -> str:
    def m(k: str) -> str:
        v = acc.get(k) or []
        return f"{mean(v):.1f} (n={len(v)})" if v else "n/a"
    return f"exams={m('exams')}  treatment={m('treatment')}  overall={m('overall')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw: list[dict] = []
    with args.csv.open(encoding="utf-8", newline="") as f:
        raw = list(csvmod.DictReader(f))
    rows = select_rows_for_l1(raw)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    print(f"sample={len(rows)} visits (seed={args.seed})", flush=True)

    import clinical_knowledge.semantic_rule_fallback as srf
    import clinical_knowledge.term_catalog as tc

    orig = srf._catalog_aliases
    print("catalog_available:", tc.catalog_available())

    print("\n[OFF] каталог синонимов выключен (базлайн)...", flush=True)
    srf._catalog_aliases = lambda term: []  # type: ignore[assignment]
    off = _run(rows, "off")
    print("OFF:", _fmt(off))

    print("\n[ON] каталог синонимов включён...", flush=True)
    srf._catalog_aliases = orig  # type: ignore[assignment]
    tc.clear_cache()
    on = _run(rows, "on")
    print("ON :", _fmt(on))

    def delta(k: str) -> str:
        a, b = off.get(k) or [], on.get(k) or []
        if not a or not b:
            return "n/a"
        return f"{mean(b) - mean(a):+.1f}"

    print("\n=== ДЕЛЬТА (ON - OFF) ===")
    print(f"exams:     {delta('exams')}")
    print(f"treatment: {delta('treatment')}")
    print(f"overall:   {delta('overall')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
