#!/usr/bin/env python3
"""Проба качества карточек-выдержек протоколов (навигатор поиска).

Для каждого gold-запроса:
- строит коды МКБ (`analyze_query_for_icd`);
- находит протоколы-кандидаты по МКБ (Summary Cards);
- сравнивает ДО (существующие nav-preview: усечённый сниппет без цитаты/страницы)
  и ПОСЛЕ (новый `protocol_card`: целые выдержки по разделам с цитатой и страницей).

Метрика: доля кандидатов со структурной выдержкой, среднее число выдержек,
доля выдержек с ссылкой на страницу и цитатой.

Запуск:
    python3 scripts/probe_protocol_cards.py \
        --golden eval/golden_icd_calibration.jsonl \
        --report-json data/ml/reports/protocol_cards_latest.json \
        --report-md data/ml/reports/protocol_cards_latest.md
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

from clinical_knowledge.protocol_summary.loader import (  # noqa: E402
    find_conditions_by_icd,
    find_summary_for_condition,
)
from clinical_knowledge.protocol_summary.nav import (  # noqa: E402
    build_protocol_card_from_summary,
    build_protocol_summary_nav,
)
from icd_mkb import analyze_query_for_icd  # noqa: E402

_MAX_CANDIDATES = 5


def _candidate_paths(query: str) -> list[str]:
    analysis = analyze_query_for_icd(query, query)
    codes = list(analysis.get("codes_for_retrieval") or [])
    paths: list[str] = []
    seen: set[str] = set()
    for code in codes:
        for cond in find_conditions_by_icd(code):
            summary = find_summary_for_condition(cond)
            if summary is None or not summary.source or not summary.source.local_path:
                continue
            p = summary.source.local_path
            if p in seen:
                continue
            seen.add(p)
            paths.append(p)
            if len(paths) >= _MAX_CANDIDATES:
                return paths
    return paths


def _baseline_previews(path: str, query: str, codes: list[str]) -> int:
    """ДО: сколько усечённых nav-preview доступно (без страницы/цитаты)."""
    nav = build_protocol_summary_nav(path, query=query, icd_codes=codes)
    if not nav.get("available"):
        return 0
    n = 0
    for cond in nav.get("conditions") or []:
        for sec in cond.get("sections") or []:
            if sec.get("preview"):
                n += 1
    return n


def evaluate(golden: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    total_candidates = 0
    baseline_with_preview = 0
    baseline_page_refs = 0
    baseline_quotes = 0
    card_available = 0
    card_ge2 = 0
    card_extracts_total = 0
    card_page_refs = 0
    card_quotes = 0

    for row in golden:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        analysis = analyze_query_for_icd(query, query)
        codes = list(analysis.get("codes_for_retrieval") or [])
        paths = _candidate_paths(query)
        cand_rows: list[dict[str, Any]] = []
        for path in paths:
            total_candidates += 1
            base_prev = _baseline_previews(path, query, codes)
            if base_prev > 0:
                baseline_with_preview += 1

            card = build_protocol_card_from_summary(path, query=query, icd_codes=codes)
            avail = bool(card.get("available"))
            extracts = card.get("extracts") or []
            n_ex = len(extracts)
            n_page = sum(1 for e in extracts if e.get("page_start"))
            n_quote = sum(1 for e in extracts if e.get("quote"))
            if avail:
                card_available += 1
                card_extracts_total += n_ex
                card_page_refs += n_page
                card_quotes += n_quote
                if n_ex >= 2:
                    card_ge2 += 1
            cand_rows.append(
                {
                    "path": Path(path).name,
                    "title": card.get("title") or "",
                    "baseline_previews": base_prev,
                    "card_available": avail,
                    "card_source": card.get("source"),
                    "n_extracts": n_ex,
                    "n_page_refs": n_page,
                    "n_quotes": n_quote,
                    "labels": [e.get("label") for e in extracts],
                }
            )
        cases.append(
            {
                "query": query,
                "predicted_codes": codes[:4],
                "n_candidates": len(paths),
                "candidates": cand_rows,
            }
        )

    def _rate(x: int) -> float:
        return round(x / total_candidates, 4) if total_candidates else 0.0

    return {
        "n_queries": len(cases),
        "n_candidates": total_candidates,
        "before": {
            "structured_coverage": _rate(baseline_with_preview),
            "page_ref_coverage": _rate(baseline_page_refs),
            "quote_coverage": _rate(baseline_quotes),
            "note": "nav-preview: усечённый сниппет 160 симв., без страницы и цитаты",
        },
        "after": {
            "structured_coverage": _rate(card_available),
            "ge2_extracts_coverage": _rate(card_ge2),
            "avg_extracts": round(card_extracts_total / card_available, 2) if card_available else 0.0,
            "page_ref_coverage": round(card_page_refs / max(1, card_extracts_total), 4),
            "quote_coverage": round(card_quotes / max(1, card_extracts_total), 4),
            "note": "protocol_card: целые выдержки по разделам с цитатой и страницей",
        },
        "cases": cases,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    b = report["before"]
    a = report["after"]
    lines: list[str] = []
    lines.append("# Проба: карточки-выдержки протоколов (навигатор)")
    lines.append("")
    lines.append(f"- Запросов: {report['n_queries']}")
    lines.append(f"- Кандидатов (протоколов): {report['n_candidates']}")
    lines.append("")
    lines.append("## ДО (nav-preview) vs ПОСЛЕ (protocol_card)")
    lines.append("")
    lines.append("| Метрика | ДО | ПОСЛЕ |")
    lines.append("|---------|----|-------|")
    lines.append(f"| Структурная выдержка (покрытие) | {b['structured_coverage']*100:.1f}% | {a['structured_coverage']*100:.1f}% |")
    lines.append(f"| ≥2 выдержек в карточке | - | {a['ge2_extracts_coverage']*100:.1f}% |")
    lines.append(f"| Ссылка на страницу | {b['page_ref_coverage']*100:.1f}% | {a['page_ref_coverage']*100:.1f}% (по выдержкам) |")
    lines.append(f"| Дословная цитата | {b['quote_coverage']*100:.1f}% | {a['quote_coverage']*100:.1f}% (по выдержкам) |")
    lines.append(f"| Ср. число выдержек | - | {a['avg_extracts']} |")
    lines.append("")
    lines.append("## Кейсы")
    lines.append("")
    for c in report["cases"]:
        lines.append(f"### {c['query']}")
        lines.append("")
        lines.append(f"МКБ: {', '.join(c['predicted_codes']) or '-'} · кандидатов: {c['n_candidates']}")
        lines.append("")
        for cand in c["candidates"]:
            labels = ", ".join(str(x) for x in cand["labels"] if x) or "-"
            lines.append(
                f"- **{cand['title'] or cand['path']}** - выдержек: {cand['n_extracts']} "
                f"(стр.: {cand['n_page_refs']}, цитат: {cand['n_quotes']}; {labels})"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_icd_calibration.jsonl")
    ap.add_argument("--report-json", default="data/ml/reports/protocol_cards_latest.json")
    ap.add_argument("--report-md", default="data/ml/reports/protocol_cards_latest.md")
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

    a = report["after"]
    b = report["before"]
    print(
        f"queries={report['n_queries']} candidates={report['n_candidates']} "
        f"| before_structured={b['structured_coverage']:.3f} "
        f"| after_structured={a['structured_coverage']:.3f} "
        f"after_ge2={a['ge2_extracts_coverage']:.3f} "
        f"after_page={a['page_ref_coverage']:.3f} after_quote={a['quote_coverage']:.3f}"
    )
    print(f"report: {jp}")
    print(f"report: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
