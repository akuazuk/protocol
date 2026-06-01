#!/usr/bin/env python3
"""Сгенерировать docs/presentation-stats.json из каталога и protocol summaries."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation-stats.json"
INDEX_CSV = ROOT / "data" / "catalog" / "index.csv"
BUILD_VERSION_FALLBACK = "2026-06-01-r57-presentation-stats-dash"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _corpus_from_index_csv() -> dict:
    if not INDEX_CSV.is_file():
        return {}
    rows = list(csv.DictReader(INDEX_CSV.read_text(encoding="utf-8").splitlines()))
    rubrics: set[str] = set()
    years: dict[str, int] = {}
    categories: dict[str, int] = {}
    post_mz = 0
    for row in rows:
        cat = (row.get("category") or row.get("rubric") or "").strip()
        if cat:
            rubrics.add(cat)
            categories[cat] = categories.get(cat, 0) + 1
        if "post" in (row.get("filename") or row.get("path") or "").lower():
            post_mz += 1
        y = (row.get("year") or row.get("years_in_filename") or "").strip()
        if y:
            years[y] = years.get(y, 0) + 1
    years_top = sorted(({"year": k, "count": v} for k, v in years.items()), key=lambda x: -x["count"])[:8]
    categories_top = sorted(
        ({"slug": k, "label": k.replace("-", " ").title(), "count": v} for k, v in categories.items()),
        key=lambda x: -x["count"],
    )[:12]
    mtime = datetime.fromtimestamp(INDEX_CSV.stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "index_csv_available": True,
        "protocols_in_index": len(rows),
        "protocols_post_mz": post_mz,
        "rubrics_in_index": len(rubrics),
        "index_csv_updated_utc": mtime,
        "years_top": years_top,
        "categories_top": categories_top,
    }


def _load_existing() -> dict:
    return _load_json(OUT)


def main() -> int:
    from clinical_knowledge import clinical_knowledge_status
    from clinical_knowledge.presentation_stats import build_presentation_stats_bundle

    existing = _load_existing()
    corpus = _corpus_from_index_csv()
    version = (existing.get("build_version") or BUILD_VERSION_FALLBACK).strip()

    ck: dict = {}
    try:
        ck = clinical_knowledge_status()
    except Exception:
        ck = existing.get("clinical_knowledge") or {}

    quality = _load_json(ROOT / "data" / "quality_benchmark.json") or existing.get("quality_benchmark")
    rag_version: dict = existing.get("rag") or {}

    bundle = build_presentation_stats_bundle(
        corpus=corpus,
        version=version,
        clinical_knowledge=ck,
        quality_benchmark=quality,
        rag_version=rag_version,
    )
    if existing.get("kz_fixtures"):
        bundle["kz_fixtures"] = existing["kz_fixtures"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pa = bundle.get("protocol_analysis") or {}
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(OUT.relative_to(ROOT)),
                "summaries": pa.get("summaries_total"),
                "rules": (bundle.get("rules_coverage") or {}).get("total_auto_rules"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
