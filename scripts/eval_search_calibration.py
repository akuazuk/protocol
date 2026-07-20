#!/usr/bin/env python3
"""Оценка качества МКБ-маршрутизации и калибровки уверенности поиска протоколов.

Для каждого gold-запроса:
- строит коды МКБ (`analyze_query_for_icd`) и проверяет top-1 / top-3 против ожидаемых префиксов;
- считает калиброванную уверенность (`calibrate_confidence`) и сверяет с фактической точностью
  (Brier score, ECE, reliability-таблица).

Запуск:
    python3 scripts/eval_search_calibration.py \
        --golden eval/golden_icd_calibration.jsonl \
        --report-json data/ml/reports/search_calibration_latest.json \
        --report-md data/ml/reports/search_calibration_latest.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.calibration_metrics import summarize_calibration  # noqa: E402
from clinical_knowledge.confidence_calibration import (  # noqa: E402
    calibrate_confidence,
    confidence_band,
)
from icd_mkb import analyze_query_for_icd  # noqa: E402

_ICD_SCORE_CAP = 25.0


def _matches_prefix(code: str, prefixes: list[str]) -> bool:
    cu = (code or "").upper().replace(" ", "")
    return any(cu.startswith(p.upper()) for p in prefixes if p)


def _top_score(analysis: dict[str, Any]) -> float:
    for s in analysis.get("suggested") or []:
        try:
            return float(s.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def evaluate(golden: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    pairs: list[tuple[float, int]] = []
    top1_hits = 0
    top3_hits = 0
    for i, row in enumerate(golden):
        query = str(row.get("query") or "").strip()
        prefixes = [str(p) for p in (row.get("expected_icd_prefixes") or [])]
        if not query or not prefixes:
            continue
        analysis = analyze_query_for_icd(query, query)
        codes = list(analysis.get("codes_for_retrieval") or [])
        top1_ok = bool(codes) and _matches_prefix(codes[0], prefixes)
        top3_ok = any(_matches_prefix(c, prefixes) for c in codes[:3])
        icd_rel = min(1.0, _top_score(analysis) / _ICD_SCORE_CAP)
        conf = calibrate_confidence(icd_relevance=icd_rel, llm_confidence=icd_rel)
        pairs.append((conf, 1 if top1_ok else 0))
        top1_hits += int(top1_ok)
        top3_hits += int(top3_ok)
        cases.append(
            {
                "query": query,
                "expected_icd_prefixes": prefixes,
                "predicted_codes": codes[:5],
                "top1_ok": top1_ok,
                "top3_ok": top3_ok,
                "confidence": conf,
                "confidence_band": confidence_band(conf),
            }
        )
    n = len(cases)
    return {
        "n": n,
        "top1_accuracy": round(top1_hits / n, 4) if n else 0.0,
        "top3_accuracy": round(top3_hits / n, 4) if n else 0.0,
        "calibration": summarize_calibration(pairs, n_bins=5),
        "cases": cases,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Отчёт: МКБ-маршрутизация и калибровка поиска")
    lines.append("")
    lines.append(f"- Кейсов: {report['n']}")
    lines.append(f"- Top-1 точность МКБ: {report['top1_accuracy'] * 100:.1f}%")
    lines.append(f"- Top-3 точность МКБ: {report['top3_accuracy'] * 100:.1f}%")
    cal = report["calibration"]
    lines.append(f"- Brier score: {cal['brier_score']} (0 - идеально)")
    lines.append(f"- ECE (ошибка калибровки): {cal['ece']} (0 - идеально)")
    lines.append(f"- Средняя уверенность: {cal['avg_confidence']}, фактическая точность: {cal['accuracy']}")
    lines.append("")
    lines.append("## Надёжность по корзинам уверенности")
    lines.append("")
    lines.append("| Диапазон | N | Ср. уверенность | Точность | Разрыв |")
    lines.append("|----------|---|-----------------|----------|--------|")
    for r in cal["reliability"]:
        lines.append(
            f"| {r['lo']}-{r['hi']} | {r['count']} | {r['avg_conf']} | {r['accuracy']} | {r['gap']} |"
        )
    lines.append("")
    lines.append("## Кейсы")
    lines.append("")
    lines.append("| Запрос | Ожидалось | Предсказано | Top-1 | Уверенность |")
    lines.append("|--------|-----------|-------------|-------|-------------|")
    for c in report["cases"]:
        q = c["query"].replace("\n", " ")[:48]
        exp = ", ".join(c["expected_icd_prefixes"][:4])
        pred = ", ".join(c["predicted_codes"][:3])
        ok = "да" if c["top1_ok"] else "нет"
        lines.append(f"| {q} | {exp} | {pred} | {ok} | {c['confidence']} ({c['confidence_band']}) |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_icd_calibration.jsonl")
    ap.add_argument("--report-json", default="data/ml/reports/search_calibration_latest.json")
    ap.add_argument("--report-md", default="data/ml/reports/search_calibration_latest.md")
    args = ap.parse_args()

    gp = ROOT / args.golden if not Path(args.golden).is_absolute() else Path(args.golden)
    golden = [
        json.loads(line)
        for line in gp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate(golden)

    jp = ROOT / args.report_json if not Path(args.report_json).is_absolute() else Path(args.report_json)
    mp = ROOT / args.report_md if not Path(args.report_md).is_absolute() else Path(args.report_md)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(_to_markdown(report), encoding="utf-8")

    print(
        f"n={report['n']} top1={report['top1_accuracy']:.3f} top3={report['top3_accuracy']:.3f} "
        f"brier={report['calibration']['brier_score']} ece={report['calibration']['ece']}"
    )
    print(f"report: {jp}")
    print(f"report: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
