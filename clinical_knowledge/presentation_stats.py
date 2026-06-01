"""Агрегаты для презентации MVP: корпус, правила, protocol summaries."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_JSON = ROOT / "data" / "protocol_summaries" / "json"
BUILD_STATE = ROOT / "data" / "catalog" / "build_state.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _protocol_summaries_analysis() -> dict[str, Any]:
    if not SUMMARIES_JSON.is_dir():
        return {"summaries_total": 0, "by_rubric": {}, "review_status": {}, "extraction_status": {}}

    by_rubric: dict[str, int] = {}
    review_status: dict[str, int] = {}
    extraction_status: dict[str, int] = {}
    conditions_total = 0
    icd10_codes: set[str] = set()
    summaries_total = 0

    for path in sorted(SUMMARIES_JSON.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        summaries_total += 1
        rubric = row.get("rubric") if isinstance(row.get("rubric"), dict) else {}
        slug = str(rubric.get("slug") or "unknown")
        by_rubric[slug] = by_rubric.get(slug, 0) + 1
        rs = str(row.get("review_status") or "unknown")
        review_status[rs] = review_status.get(rs, 0) + 1
        es = str(row.get("extraction_status") or "unknown")
        extraction_status[es] = extraction_status.get(es, 0) + 1
        conds = row.get("conditions")
        if isinstance(conds, list):
            conditions_total += len(conds)
            for c in conds:
                if not isinstance(c, dict):
                    continue
                codes = c.get("icd10_codes")
                if isinstance(codes, list):
                    for code in codes:
                        if code:
                            icd10_codes.add(str(code).strip().upper())

    top_rubrics = sorted(
        (
            {"slug": slug, "count": cnt}
            for slug, cnt in by_rubric.items()
        ),
        key=lambda x: (-x["count"], x["slug"]),
    )[:15]

    return {
        "summaries_total": summaries_total,
        "conditions_in_summaries": conditions_total,
        "unique_icd10_in_summaries": len(icd10_codes),
        "by_rubric": by_rubric,
        "top_rubrics": top_rubrics,
        "review_status": review_status,
        "extraction_status": extraction_status,
    }


def _top_rules_by_condition(rules_by_condition: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    items: list[tuple[str, int]] = []
    for k, v in (rules_by_condition or {}).items():
        try:
            items.append((str(k), int(v)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda x: (-x[1], x[0]))
    return [{"condition_id": k, "rules": n} for k, n in items[:limit]]


def build_presentation_stats_bundle(
    *,
    corpus: dict[str, Any] | None = None,
    version: str | None = None,
    clinical_knowledge: dict[str, Any] | None = None,
    quality_benchmark: dict[str, Any] | None = None,
    rag_version: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать JSON для docs/presentation-stats.json и /api/presentation-stats."""
    from clinical_knowledge.coverage import coverage_status_payload

    build_state = _load_json(BUILD_STATE)
    cov = coverage_status_payload()
    summaries = _protocol_summaries_analysis()
    rules_by_condition = cov.get("rules_by_condition") or {}
    by_rubric = cov.get("by_rubric") or {}

    rubric_rows = []
    for slug, row in sorted(by_rubric.items(), key=lambda x: (-(x[1].get("pdfs_total") or 0), x[0])):
        if not isinstance(row, dict):
            continue
        rubric_rows.append(
            {
                "slug": slug,
                "pdfs_total": row.get("pdfs_total"),
                "pdfs_with_rules": row.get("pdfs_with_rules"),
                "coverage_pct": row.get("coverage_pct"),
            }
        )

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "build_version": version,
        "corpus": corpus or {},
        "clinical_knowledge": clinical_knowledge or {},
        "catalog_build": build_state,
        "quality_benchmark": quality_benchmark,
        "rag": rag_version or {},
        "rules_coverage": {
            "pdfs_total": cov.get("pdfs_total"),
            "pdfs_with_rules": cov.get("pdfs_with_rules"),
            "coverage_pct": cov.get("coverage_pct"),
            "total_auto_rules": cov.get("total_auto_rules"),
            "rules_by_condition": rules_by_condition,
            "top_rules_by_condition": _top_rules_by_condition(rules_by_condition),
            "by_rubric": rubric_rows,
        },
        "protocol_analysis": summaries,
    }
