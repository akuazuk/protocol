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
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.page_locator import locate_page_for_quote  # noqa: E402
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
_RICH_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"


def _path_key(path: str) -> str:
    base = Path(str(path or "")).name.lower()
    base = re.sub(r"\.pdf$", "", base)
    return re.sub(r"[^a-zа-я0-9]", "", base)


def _build_chunk_index(keys: set[str]) -> dict[str, list[dict[str, Any]]]:
    """key(basename) -> список чанков {text, page_from} для указанных протоколов."""
    index: dict[str, list[dict[str, Any]]] = {}
    if not _RICH_CHUNKS.is_file() or not keys:
        return index
    with _RICH_CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            key = _path_key(d.get("source_path") or d.get("file_name") or "")
            if key not in keys:
                continue
            index.setdefault(key, []).append(
                {"text": d.get("text") or "", "page_from": d.get("page_from")}
            )
    return index


def _make_page_lookup(index: dict[str, list[dict[str, Any]]]):
    def _lookup(path: str, quote: str) -> int | None:
        chunks = index.get(_path_key(path))
        if not chunks:
            return None
        return locate_page_for_quote(quote, chunks)

    return _lookup


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
    # 1-й проход: собрать пути-кандидаты, чтобы проиндексировать только их чанки.
    per_query_paths: list[tuple[str, list[str], list[str]]] = []
    all_keys: set[str] = set()
    for row in golden:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        codes = list(analyze_query_for_icd(query, query).get("codes_for_retrieval") or [])
        paths = _candidate_paths(query)
        per_query_paths.append((query, codes, paths))
        for p in paths:
            all_keys.add(_path_key(p))

    chunk_index = _build_chunk_index(all_keys)
    page_lookup = _make_page_lookup(chunk_index)

    cases: list[dict[str, Any]] = []
    total_candidates = 0
    baseline_with_preview = 0
    card_available = 0
    card_ge2 = 0
    card_extracts_total = 0
    card_quotes = 0
    page_summary = 0
    page_matched = 0

    for query, codes, paths in per_query_paths:
        cand_rows: list[dict[str, Any]] = []
        for path in paths:
            total_candidates += 1
            base_prev = _baseline_previews(path, query, codes)
            if base_prev > 0:
                baseline_with_preview += 1

            card = build_protocol_card_from_summary(
                path, query=query, icd_codes=codes, page_lookup=page_lookup
            )
            avail = bool(card.get("available"))
            extracts = card.get("extracts") or []
            n_ex = len(extracts)
            n_quote = sum(1 for e in extracts if e.get("quote"))
            n_page_summary = sum(1 for e in extracts if e.get("page_source") == "summary")
            n_page_matched = sum(1 for e in extracts if e.get("page_source") == "matched")
            if avail:
                card_available += 1
                card_extracts_total += n_ex
                card_quotes += n_quote
                page_summary += n_page_summary
                page_matched += n_page_matched
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
                    "n_page_summary": n_page_summary,
                    "n_page_matched": n_page_matched,
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

    ex_total = max(1, card_extracts_total)
    return {
        "n_queries": len(cases),
        "n_candidates": total_candidates,
        "chunk_index_protocols": len(chunk_index),
        "before": {
            "structured_coverage": _rate(baseline_with_preview),
            "note": "nav-preview: усечённый сниппет 160 симв., без страницы и цитаты",
        },
        "after": {
            "structured_coverage": _rate(card_available),
            "ge2_extracts_coverage": _rate(card_ge2),
            "avg_extracts": round(card_extracts_total / card_available, 2) if card_available else 0.0,
            "quote_coverage": round(card_quotes / ex_total, 4),
            "note": "protocol_card: целые выдержки по разделам с цитатой и страницей",
        },
        "page_enrichment": {
            "extracts_total": card_extracts_total,
            "page_before": round(page_summary / ex_total, 4),
            "page_after": round((page_summary + page_matched) / ex_total, 4),
            "matched_added": page_matched,
            "note": "page_before - страница из карточки; page_after - плюс сопоставление цитаты с чанком",
        },
        "cases": cases,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    b = report["before"]
    a = report["after"]
    pe = report["page_enrichment"]
    lines: list[str] = []
    lines.append("# Проба: карточки-выдержки протоколов (навигатор)")
    lines.append("")
    lines.append(f"- Запросов: {report['n_queries']}")
    lines.append(f"- Кандидатов (протоколов): {report['n_candidates']}")
    lines.append(f"- Проиндексировано протоколов для сопоставления страниц: {report.get('chunk_index_protocols', 0)}")
    lines.append("")
    lines.append("## ДО (nav-preview) vs ПОСЛЕ (protocol_card)")
    lines.append("")
    lines.append("| Метрика | ДО | ПОСЛЕ |")
    lines.append("|---------|----|-------|")
    lines.append(f"| Структурная выдержка (покрытие) | {b['structured_coverage']*100:.1f}% | {a['structured_coverage']*100:.1f}% |")
    lines.append(f"| ≥2 выдержек в карточке | - | {a['ge2_extracts_coverage']*100:.1f}% |")
    lines.append(f"| Дословная цитата (по выдержкам) | - | {a['quote_coverage']*100:.1f}% |")
    lines.append(f"| Ср. число выдержек | - | {a['avg_extracts']} |")
    lines.append("")
    lines.append("## Обогащение страниц (сопоставление цитаты с чанком)")
    lines.append("")
    lines.append("| Ссылка на страницу (по выдержкам) | ДО | ПОСЛЕ |")
    lines.append("|-----------------------------------|----|-------|")
    lines.append(f"| Покрытие | {pe['page_before']*100:.1f}% | {pe['page_after']*100:.1f}% |")
    lines.append(f"| Добавлено страниц сопоставлением | - | {pe['matched_added']} из {pe['extracts_total']} выдержек |")
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
            pages = cand["n_page_summary"] + cand["n_page_matched"]
            lines.append(
                f"- **{cand['title'] or cand['path']}** - выдержек: {cand['n_extracts']} "
                f"(стр.: {pages} [карточка {cand['n_page_summary']} + сопост. {cand['n_page_matched']}], "
                f"цитат: {cand['n_quotes']}; {labels})"
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
    pe = report["page_enrichment"]
    print(
        f"queries={report['n_queries']} candidates={report['n_candidates']} "
        f"| structured {b['structured_coverage']:.3f}->{a['structured_coverage']:.3f} "
        f"| after_ge2={a['ge2_extracts_coverage']:.3f} after_quote={a['quote_coverage']:.3f} "
        f"| page {pe['page_before']:.3f}->{pe['page_after']:.3f} (+{pe['matched_added']})"
    )
    print(f"report: {jp}")
    print(f"report: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
