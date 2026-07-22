#!/usr/bin/env python3
"""Э4.3-4.4: калибровка порогов статусов для axes-режима (overall по alignment-блокам).

Проблема (из Э3.2): alignment даёт более строгую (реалистичную) оценку покрытия, чем
прежний rules-based блок; при дефолтных порогах 90/75/50 включение осей заливает
non_compliant (-17.5 к overall). Нужны свои пороги для нового распределения.

Подход - distribution-preserving калибровка: на gold-выборке считаем распределение
axes-overall и baseline-статусов, затем ставим пороги так, чтобы доли статусов в
axes-режиме совпали с baseline (стабильный flag-rate, но лучшая нацеленность). Если
есть LLM-метки (gold_llm_labels.jsonl) - дополнительно считаем QWK(статус vs LLM)
до/после как sanity-check.

  PYTHONPATH=. python3 scripts/calibrate_axes_thresholds.py \\
    --gold /var/data/mis_protocol/kz_gold/gold_sample.jsonl \\
    --csv  /var/data/mis_protocol/mis_protocol_2026-07.csv \\
    --labels /var/data/mis_protocol/kz_gold/gold_llm_labels.jsonl \\
    --out-config config/axes_thresholds.yaml
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "clinical_knowledge").is_dir():
    ROOT = Path(os.environ.get("PROTOCOL_ROOT") or "/opt/render/project/src")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mis_protocol_l1_batch import (  # noqa: E402
    _direct_tier,
    build_kz_text,
)

# Порядок градуированных статусов (для квантилей и QWK).
_RANK = {"non_compliant": 0, "partially_compliant": 1, "mostly_compliant": 2, "compliant": 3}
_GRADED = set(_RANK)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _qwk(pairs: list[tuple[int, int]], n_cls: int = 4) -> float | None:
    """Quadratic weighted kappa для пар (pred, true) ординальных рангов."""
    if not pairs:
        return None
    O = [[0] * n_cls for _ in range(n_cls)]
    for a, b in pairs:
        O[a][b] += 1
    row = [sum(O[i]) for i in range(n_cls)]
    col = [sum(O[i][j] for i in range(n_cls)) for j in range(n_cls)]
    n = len(pairs)
    num = den = 0.0
    for i in range(n_cls):
        for j in range(n_cls):
            w = ((i - j) ** 2) / ((n_cls - 1) ** 2)
            e = row[i] * col[j] / n
            num += w * O[i][j]
            den += w * e
    if den == 0:
        return None
    return round(1 - num / den, 3)


def _status_from_thr(score: float, thr: dict) -> str:
    if score >= thr["compliant"]:
        return "compliant"
    if score >= thr["mostly_compliant"]:
        return "mostly_compliant"
    if score >= thr["partially_compliant"]:
        return "partially_compliant"
    return "non_compliant"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out-config", type=Path, default=ROOT / "config" / "axes_thresholds.yaml")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "ml" / "kz_gold" / "calibration_report.md")
    args = ap.parse_args()

    gold_ids: list[str] = []
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                gold_ids.append(str(json.loads(line).get("visit_id") or ""))
            except json.JSONDecodeError:
                pass
    gold_ids = [v for v in gold_ids if v]

    with args.csv.open(encoding="utf-8", newline="") as f:
        csv_by_visit = {str(r.get("visit_id") or ""): r for r in csvmod.DictReader(f)}

    import clinical_knowledge.semantic_rule_fallback as srf  # noqa: F401
    import clinical_knowledge.term_catalog as tc

    def set_cfg(axes: bool) -> None:
        os.environ["CONSULT_STRUCTURED_ITEMS"] = "1"
        os.environ["CONSULT_AXES_OVERALL"] = "1" if axes else "0"
        tc.clear_cache()

    # baseline (axes OFF) и axes (ON) в одном проходе.
    recs: list[dict] = []
    print(f"gold={len(gold_ids)} визитов; считаю baseline + axes...", flush=True)
    for i, vid in enumerate(gold_ids, 1):
        row = csv_by_visit.get(vid)
        if not row:
            continue
        text = build_kz_text(row)
        try:
            set_cfg(False)
            b = _direct_tier(text, f"mis-calib-b-{vid}")
            set_cfg(True)
            a = _direct_tier(text, f"mis-calib-a-{vid}")
        except Exception:
            continue
        recs.append({
            "visit_id": vid,
            "base_status": str(b.get("overall_status") or ""),
            "base_overall": b.get("overall_score"),
            "axes_status": str(a.get("overall_status") or ""),
            "axes_overall": a.get("overall_score"),
        })
        if i % 50 == 0:
            print(f"  ... {i}/{len(gold_ids)}", flush=True)

    # Распределение baseline-статусов (только градуированные).
    base_graded = [r["base_status"] for r in recs if r["base_status"] in _GRADED]
    base_dist = Counter(base_graded)
    n_base = len(base_graded) or 1
    frac = {s: base_dist.get(s, 0) / n_base for s in _RANK}

    # axes-overall по градуированным baseline-кейсам (без manual/insufficient).
    axes_scores = sorted(
        float(r["axes_overall"]) for r in recs
        if r["base_status"] in _GRADED and isinstance(r["axes_overall"], (int, float))
    )

    cum_non = frac["non_compliant"]
    cum_partial = cum_non + frac["partially_compliant"]
    cum_mostly = cum_partial + frac["mostly_compliant"]

    thr = {
        "compliant": round(_quantile(axes_scores, cum_mostly), 1),
        "mostly_compliant": round(_quantile(axes_scores, cum_partial), 1),
        "partially_compliant": round(_quantile(axes_scores, cum_non), 1),
        "non_compliant": 1,
    }
    # Гарантируем монотонность/разумность.
    thr["mostly_compliant"] = min(thr["mostly_compliant"], thr["compliant"])
    thr["partially_compliant"] = min(thr["partially_compliant"], thr["mostly_compliant"])

    # Валидация: доли статусов после калибровки vs baseline.
    calib_dist: Counter = Counter()
    for r in recs:
        if r["base_status"] not in _GRADED:
            continue
        sc = r["axes_overall"]
        if isinstance(sc, (int, float)):
            calib_dist[_status_from_thr(float(sc), thr)] += 1

    # LLM-метки (если есть) → QWK.
    llm: dict[str, str] = {}
    if args.labels and args.labels.is_file():
        for line in args.labels.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = str(rec.get("llm_status") or "")
            if st in _GRADED and not rec.get("error"):
                llm[str(rec.get("visit_id") or "")] = st

    qwk_base = qwk_calib = None
    n_labeled = 0
    if llm:
        pb: list[tuple[int, int]] = []
        pc: list[tuple[int, int]] = []
        for r in recs:
            vid = r["visit_id"]
            if vid not in llm:
                continue
            t = _RANK[llm[vid]]
            if r["base_status"] in _GRADED:
                pb.append((_RANK[r["base_status"]], t))
            sc = r["axes_overall"]
            if isinstance(sc, (int, float)):
                pc.append((_RANK[_status_from_thr(float(sc), thr)], t))
        n_labeled = len(pc)
        qwk_base = _qwk(pb)
        qwk_calib = _qwk(pc)

    # Пишем config.
    ver = "1.0"
    lines_cfg = [
        "# Калиброванные пороги статусов для axes-режима (Э4).",
        "# distribution-preserving: доли статусов ≈ baseline; включается при CONSULT_AXES_OVERALL=1.",
        f"version: '{ver}'",
        f"calibrated_on_gold_n: {len(recs)}",
        "status_thresholds:",
        f"  compliant: {thr['compliant']}",
        f"  mostly_compliant: {thr['mostly_compliant']}",
        f"  partially_compliant: {thr['partially_compliant']}",
        f"  non_compliant: {thr['non_compliant']}",
    ]
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text("\n".join(lines_cfg) + "\n", encoding="utf-8")

    def _top(c: Counter) -> str:
        tot = sum(c.values()) or 1
        return ", ".join(f"{k}:{v}({v/tot:.0%})" for k, v in sorted(c.items(), key=lambda x: -_RANK.get(x[0], -1)))

    rep = [
        "# Калибровка порогов axes-overall (Э4)",
        "",
        f"gold: {len(recs)} визитов с оценкой  |  градуированных baseline: {n_base}",
        "",
        "## Пороги (distribution-preserving)",
        f"- compliant ≥ **{thr['compliant']}**",
        f"- mostly_compliant ≥ **{thr['mostly_compliant']}**",
        f"- partially_compliant ≥ **{thr['partially_compliant']}**",
        f"- non_compliant ≥ {thr['non_compliant']}",
        "",
        "## Распределение статусов (градуированные)",
        f"- baseline:        {_top(base_dist)}",
        f"- axes+калибровка: {_top(calib_dist)}",
        "",
        "## Согласие с LLM-метками (QWK, чем выше тем лучше)",
    ]
    if qwk_calib is not None:
        rep += [
            f"- размечено LLM: {n_labeled}",
            f"- QWK baseline vs LLM:        **{qwk_base}**",
            f"- QWK axes+калибровка vs LLM: **{qwk_calib}**",
        ]
    else:
        rep += ["- LLM-меток пока нет/мало (батч ещё идёт) - QWK не считался."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("\n=== КАЛИБРОВКА ===")
    print("пороги:", thr)
    print("baseline dist:  ", _top(base_dist))
    print("axes+calib dist:", _top(calib_dist))
    if qwk_calib is not None:
        print(f"QWK baseline={qwk_base}  axes+calib={qwk_calib}  (n={n_labeled})")
    print(f"config -> {args.out_config}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
