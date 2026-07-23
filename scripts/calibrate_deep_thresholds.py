#!/usr/bin/env python3
"""Э4.3/4.4: калибровка порогов deep-оценки КЗ и валидация против LLM-прокси.

Джойнит LLM-предразметку gold (llm_overall_pct / llm_status) со СВЕЖИМ выходом
deep-движка (axes + overall_pct + severity) по visit_id и:

- пересчитывает канонический deep-статус из overall+severity (risk-gate);
- считает согласованность deep vs LLM: corr(overall), QWK(бэнды), recall/precision
  детекции "плохих" КЗ (LLM overall<60 или non/partially_compliant);
- подбирает пороги статусов (good/acceptable) и порог harm-flag по deep_overall,
  плюс правило min-axis (слабая ось не должна маскироваться средним);
- пишет config/deep_thresholds.yaml (потребляется kz_deep_eval) и отчёт .md
  с метриками ДО/ПОСЛЕ калибровки.

  PYTHONPATH=. python3 scripts/calibrate_deep_thresholds.py \\
    --labels data/ml/kz_gold/gold_llm_labels_2026-01.jsonl \\
    --cases  data/ml/reports/deep_eval/kz_l1_2026-01_cases.jsonl \\
    --out-config config/deep_thresholds.yaml \\
    --report data/ml/reports/deep_calibration_2026-01.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Каноническая risk-gate логика (зеркалит kz_deep_eval._apply_risk_gate),
# параметризованная порогами для калибровки.
SEV_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
LLM_RANK = {"non_compliant": 0, "partially_compliant": 1, "mostly_compliant": 2, "compliant": 3}


def deep_status(overall, worst_sev, t_good=80.0, t_acc=60.0,
                axes=None, min_axis=None):
    if overall is None:
        return "insufficient_data"
    if worst_sev == 0:
        return "critical"
    if worst_sev == 1:
        return "review" if min(overall, 60.0) >= 50 else "poor"
    # правило слабой оси: если любая присутствующая ось ниже min_axis -> не выше review
    if min_axis is not None and axes:
        present = [v for v in axes.values() if isinstance(v, (int, float))]
        if present and min(present) < min_axis:
            return "review"
    if overall >= t_good:
        return "good"
    if overall >= t_acc:
        return "acceptable"
    return "review"


def band(pct):
    if pct is None:
        return None
    if pct < 50:
        return 0
    if pct < 70:
        return 1
    if pct < 80:
        return 2
    return 3


def qwk(a, b, k=4):
    """Quadratic weighted kappa для целочисленных бэндов 0..k-1."""
    import numpy as np
    a = np.asarray(a); b = np.asarray(b)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 / (k - 1) ** 2
    act = O.sum(axis=1); pred = O.sum(axis=0); N = O.sum()
    E = np.outer(act, pred) / max(N, 1)
    num = (W * O).sum(); den = (W * E).sum()
    return 1.0 - num / den if den else 0.0


def corr(xs, ys):
    if len(xs) < 3:
        return None
    try:
        return round(st.correlation(xs, ys), 3)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--out-config", type=Path, default=Path("config/deep_thresholds.yaml"))
    ap.add_argument("--report", type=Path, default=Path("data/ml/reports/deep_calibration.md"))
    args = ap.parse_args()

    # deep по visit_id
    deep_by_vid = {}
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r.get("deep") or {}
        if d:
            deep_by_vid[str(r.get("visit_id"))] = d

    rows = []
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lab = json.loads(line)
        except json.JSONDecodeError:
            continue
        if lab.get("error") or lab.get("llm_overall_pct") is None:
            continue
        vid = str(lab.get("visit_id"))
        d = deep_by_vid.get(vid)
        if not d or d.get("overall_pct") is None:
            continue
        sev = d.get("n_by_severity") or {}
        worst = 9
        for s in ("P0", "P1", "P2", "P3"):
            if sev.get(s):
                worst = SEV_ORDER[s]; break
        rows.append({
            "vid": vid,
            "deep_overall": float(d["overall_pct"]),
            "axes": d.get("axes") or {},
            "worst": worst,
            "llm_overall": float(lab["llm_overall_pct"]),
            "llm_status": lab.get("llm_status"),
            "llm_rank": LLM_RANK.get(lab.get("llm_status"), None),
            "n_crit": lab.get("n_critical_gaps") or 0,
        })

    n = len(rows)
    if n < 10:
        print(f"too few joined rows: {n} (labels still running?)")
        return 1

    deep_o = [r["deep_overall"] for r in rows]
    llm_o = [r["llm_overall"] for r in rows]
    # "плохой" КЗ по LLM
    llm_bad = [1 if (r["llm_overall"] < 60 or (r["llm_rank"] is not None and r["llm_rank"] <= 1)) else 0 for r in rows]
    n_bad = sum(llm_bad)

    def eval_cfg(t_good, t_acc, flag_cut, min_axis):
        ds = [deep_status(r["deep_overall"], r["worst"], t_good, t_acc, r["axes"], min_axis) for r in rows]
        deep_flag = [1 if s in ("review", "poor", "critical") else 0 for s in ds]
        # доп. flag по порогу overall
        deep_flag = [1 if (deep_flag[i] or rows[i]["deep_overall"] < flag_cut) else 0 for i in range(n)]
        tp = sum(1 for i in range(n) if deep_flag[i] and llm_bad[i])
        fp = sum(1 for i in range(n) if deep_flag[i] and not llm_bad[i])
        fn = sum(1 for i in range(n) if not deep_flag[i] and llm_bad[i])
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        # QWK по бэндам deep vs llm
        db = [band(r["deep_overall"]) for r in rows]
        lb = [band(r["llm_overall"]) for r in rows]
        k = qwk(db, lb)
        return {"recall": recall, "prec": prec, "f1": f1, "qwk": round(k, 3),
                "flag_rate": round(sum(deep_flag) / n, 3)}

    baseline = eval_cfg(80.0, 60.0, 0.0, None)

    # поиск: максимизируем harm-recall при precision>=0.45, tie-break f1
    best = None
    grid = []
    for t_good in (78, 80, 82, 85):
        for t_acc in (58, 60, 62, 65, 68, 70):
            for flag_cut in (0, 60, 65, 70, 72):
                for min_axis in (None, 50, 55, 60):
                    m = eval_cfg(t_good, t_acc, flag_cut, min_axis)
                    m.update(t_good=t_good, t_acc=t_acc, flag_cut=flag_cut, min_axis=min_axis)
                    grid.append(m)
                    ok = m["prec"] >= 0.45
                    key = (m["recall"] + 0.5 * m["f1"] + 0.3 * m["qwk"]) if ok else -1
                    if best is None or key > best[0]:
                        best = (key, m)
    best_cfg = best[1]
    after = {k: best_cfg[k] for k in ("recall", "prec", "f1", "qwk", "flag_rate")}

    corr_o = corr(deep_o, llm_o)
    mae = round(sum(abs(a - b) for a, b in zip(deep_o, llm_o)) / n, 1)
    # покрытие осей корреляцией с LLM (для рекомендации весов)
    axis_corr = {}
    for ax in ("documentation", "clinical_concordance", "safety", "regulatory"):
        xs = [r["axes"].get(ax) for r in rows]
        pair = [(x, r["llm_overall"]) for x, r in zip(xs, rows) if isinstance(x, (int, float))]
        if len(pair) >= 5:
            axis_corr[ax] = corr([p[0] for p in pair], [p[1] for p in pair])

    # config
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    cfg_lines = [
        "# Калибровка deep-порогов КЗ (Э4). Источник: gold + LLM-прокси (proxy, не методист).",
        f"# n={n}, LLM-bad={n_bad}, corr(overall)={corr_o}, MAE={mae}",
        "status_thresholds:",
        f"  good: {best_cfg['t_good']}",
        f"  acceptable: {best_cfg['t_acc']}",
        f"harm_flag_overall_cutoff: {best_cfg['flag_cut']}",
        f"min_axis_review: {best_cfg['min_axis'] if best_cfg['min_axis'] is not None else 'null'}",
        "# равные веса осей по умолчанию; корреляции осей с LLM для справки:",
    ]
    for ax, c in axis_corr.items():
        cfg_lines.append(f"#   {ax}: corr={c}")
    args.out_config.write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")

    # report
    rep = [
        f"# Калибровка/валидация deep-оценки КЗ (2026-01, proxy-LLM)\n",
        f"Join: **n={n}** размеченных gold-КЗ (LLM overall не пуст), LLM-bad={n_bad} ({round(100*n_bad/n,1)}%).\n",
        f"- corr(deep_overall, llm_overall) = **{corr_o}**, MAE = **{mae}** п.п.\n",
        "## Детекция плохих КЗ (deep review/poor/critical или overall<cutoff) vs LLM-bad\n",
        "| конфиг | good | acc | flag_cut | min_axis | recall | prec | F1 | QWK | flag_rate |",
        "|--|--|--|--|--|--|--|--|--|--|",
        f"| baseline | 80 | 60 | - | - | {baseline['recall']:.2f} | {baseline['prec']:.2f} | {baseline['f1']:.2f} | {baseline['qwk']} | {baseline['flag_rate']} |",
        f"| **калибр.** | {best_cfg['t_good']} | {best_cfg['t_acc']} | {best_cfg['flag_cut']} | {best_cfg['min_axis']} | {after['recall']:.2f} | {after['prec']:.2f} | {after['f1']:.2f} | {after['qwk']} | {after['flag_rate']} |",
        "\n## Корреляция осей с LLM-overall (для будущих весов)\n",
        "| ось | corr |", "|--|--|",
    ]
    for ax, c in axis_corr.items():
        rep.append(f"| {ax} | {c} |")
    rep.append(f"\nКонфиг записан: `{args.out_config}`.\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"n={n} bad={n_bad} corr={corr_o} MAE={mae}")
    print("baseline:", baseline)
    print("calibrated:", best_cfg)
    print("axis_corr:", axis_corr)
    print("wrote", args.out_config, "and", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
