#!/usr/bin/env python3
"""Before/after бенчмарк applicability-gate (ТЗ №2 §13, метрика «Неверная population в Top-1»).

BEFORE - ранжирование только по score (как раньше): считаем случаи, когда для
взрослого/неопределённого запроса Top-1 - детский протокол.
AFTER  - тот же набор через apply_applicability_gate.

Запуск:
    python scripts/bench_search_applicability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.search_applicability_gate import (  # noqa: E402
    apply_applicability_gate,
)

FIXTURE = ROOT / "tests" / "fixtures" / "search_applicability_golden.jsonl"
CHILD = {"child", "children", "pediatric"}


def _rows() -> list[dict]:
    out = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def _audience(row: dict) -> str:
    return (row.get("patient") or {}).get("adult_or_child", "")


def main() -> int:
    rows = _rows()
    before_bad = 0
    after_bad = 0
    before_reco_child = 0
    after_reco_child = 0
    considered = 0

    for row in rows:
        aud = _audience(row)
        # интересует «взрослый или неопределённый» запрос (не подтверждённо детский)
        if aud in ("child", "newborn"):
            continue
        considered += 1
        cands = row["candidates"]
        # BEFORE: сорт только по score
        before = sorted(cands, key=lambda c: -float(c.get("match_score") or 0))
        from clinical_knowledge.search_applicability_gate import infer_card_population
        if infer_card_population(before[0]) in CHILD:
            before_bad += 1
        # BEFORE recommended-child (наивно: всё population-specific со score>=60 «Рекомендуем»)
        for c in cands:
            if infer_card_population(c) in CHILD and float(c.get("match_score") or 0) >= 60:
                before_reco_child += 1
                break
        # AFTER: gate
        ped = aud in ("child", "newborn")
        gated = apply_applicability_gate(
            cands, row.get("patient") or {}, row.get("icd_query") or [],
            pediatric_signal=ped, keep_not_applicable=True,
        )
        if gated and gated[0]["_gate"]["population"] in CHILD:
            after_bad += 1
        for item in gated:
            g = item["_gate"]
            if g["population"] in CHILD and g["recommended"]:
                after_reco_child += 1
                break

    print("== applicability gate benchmark (adult/unknown queries) ==")
    print(f"queries considered:               {considered}")
    print(f"BEFORE invalid child Top-1:       {before_bad}")
    print(f"AFTER  invalid child Top-1:       {after_bad}   (target 0)")
    print(f"BEFORE recommended child (naive): {before_reco_child}")
    print(f"AFTER  recommended child:         {after_reco_child}   (target 0)")
    ok = after_bad == 0 and after_reco_child == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
