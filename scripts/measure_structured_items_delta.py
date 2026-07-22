#!/usr/bin/env python3
"""Э3 верификация: дельта exams/treatment от структурных items (+ каталог синонимов).

Один процесс, одна выборка, три конфигурации L1:
  1) baseline      - CONSULT_STRUCTURED_ITEMS=0, каталог OFF (как на проде сейчас)
  2) structured    - структурные required_exams/treatment ON, каталог OFF
  3) struct+catalog- структурные items ON + каталог синонимов ON

  PYTHONPATH=. python3 scripts/measure_structured_items_delta.py \\
    --csv /var/data/mis_protocol/mis_protocol_2026-07.csv --limit 300
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
from scripts.measure_semantic_match_delta import _block_scores  # noqa: E402


def _run(rows: list[dict], tag: str) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {"exams": [], "treatment": [], "overall": []}
    for i, row in enumerate(rows, 1):
        vid = str(row.get("visit_id") or row.get("id") or "")
        try:
            res = _direct_tier(build_kz_text(row), f"mis-{tag}-{vid}")
        except Exception:
            continue
        bs = _block_scores(res)
        for k in ("exams", "treatment"):
            if k in bs:
                acc[k].append(bs[k])
        ov = res.get("overall_score")
        if isinstance(ov, (int, float)):
            acc["overall"].append(float(ov))
        if i % 100 == 0:
            print(f"  ... {i}/{len(rows)}", flush=True)
    return acc


def _fmt(acc: dict[str, list[float]]) -> str:
    def m(k: str) -> str:
        v = acc.get(k) or []
        return f"{mean(v):.1f}(n={len(v)})" if v else "n/a"
    return f"exams={m('exams')}  treatment={m('treatment')}  overall={m('overall')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as f:
        raw = list(csvmod.DictReader(f))
    rows = select_rows_for_l1(raw)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    print(f"sample={len(rows)} visits (seed={args.seed})", flush=True)

    import clinical_knowledge.semantic_rule_fallback as srf
    import clinical_knowledge.term_catalog as tc
    orig = srf._catalog_aliases

    def set_cfg(structured: bool, catalog: bool) -> None:
        os.environ["CONSULT_STRUCTURED_ITEMS"] = "1" if structured else "0"
        srf._catalog_aliases = orig if catalog else (lambda term: [])  # type: ignore[assignment]
        tc.clear_cache()

    print("catalog_available:", tc.catalog_available())

    print("\n[1] baseline (structured OFF, catalog OFF)...", flush=True)
    set_cfg(False, False)
    base = _run(rows, "base")
    print("baseline:      ", _fmt(base))

    print("\n[2] structured items ON (catalog OFF)...", flush=True)
    set_cfg(True, False)
    struct = _run(rows, "struct")
    print("structured:    ", _fmt(struct))

    print("\n[3] structured + catalog ON...", flush=True)
    set_cfg(True, True)
    both = _run(rows, "both")
    print("struct+catalog:", _fmt(both))

    def d(a: dict, b: dict, k: str) -> str:
        x, y = a.get(k) or [], b.get(k) or []
        return f"{mean(y) - mean(x):+.1f}" if x and y else "n/a"

    print("\n=== ДЕЛЬТА к baseline ===")
    for label, cfg in (("structured", struct), ("struct+catalog", both)):
        print(f"{label:16s} exams {d(base,cfg,'exams')}  treatment {d(base,cfg,'treatment')}  overall {d(base,cfg,'overall')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
