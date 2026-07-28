"""Публичная аналитика пилота для главной (без ПДн, без METHODIST_TOKEN)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compliance_buckets(kz_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Распределение overall % по последнему прогону на text_hash."""
    by_hash: dict[str, float | None] = {}
    for ev in sorted(kz_events, key=lambda r: r.get("ts") or ""):
        th = ev.get("text_hash") or ""
        if not th:
            continue
        val = ev.get("compliance_overall_pct")
        by_hash[th] = float(val) if isinstance(val, (int, float)) else None
    buckets = [
        ("≥85%", 85, 101),
        ("70-84%", 70, 85),
        ("<70% (риск)", 0, 70),
        ("нет %", None, None),
    ]
    counts = {label: 0 for label, _, _ in buckets}
    for pct in by_hash.values():
        if pct is None:
            counts["нет %"] += 1
            continue
        placed = False
        for label, lo, hi in buckets:
            if lo is None:
                continue
            if lo <= pct < hi:
                counts[label] += 1
                placed = True
                break
        if not placed:
            counts["нет %"] += 1
    return [{"label": label, "count": counts[label]} for label, _, _ in buckets if counts[label] or label != "нет %"]


def build_public_pilot_analytics(
    *,
    corpus: dict[str, Any] | None = None,
    version: str | None = None,
    rag_ready: bool | None = None,
) -> dict[str, Any]:
    """Агрегаты из feedback + корпус; без analysis_id, text_hash, reviewer."""
    from clinical_knowledge.methodist_stats import build_methodist_dashboard_stats

    try:
        stats = build_methodist_dashboard_stats()
    except Exception:
        stats = {}

    summary = stats.get("summary") or {}
    charts = stats.get("charts") or {}
    pool = stats.get("pool") or {}
    ml = stats.get("ml_readiness") or {}
    protocol_match = stats.get("protocol_match") or {}

    kz_total = int(summary.get("total_kz_runs") or 0)
    reviews = int(summary.get("analysis_reviews") or 0)
    unique_kz = int(summary.get("unique_kz") or 0)
    has_real = kz_total > 0 or reviews > 0

    avg_pct = None
    risk_rate = None
    weak_blocks: list[dict[str, Any]] = []
    if stats.get("block_overrides_top"):
        for row in stats.get("block_overrides_top") or []:
            weak_blocks.append({"name": row.get("block_key") or "?", "avg_pct": None, "count": row.get("count")})

    # Средний % и зона риска из kz_analysis (если есть)
    from clinical_knowledge.methodist_stats import iter_feedback_events
    from clinical_knowledge.feedback_store import feedback_dir as resolve_feedback_dir

    kz_events = [e for e in iter_feedback_events(resolve_feedback_dir()) if e.get("event_type") == "kz_analysis"]
    pcts: list[float] = []
    risk_n = 0
    seen: set[str] = set()
    for ev in sorted(kz_events, key=lambda r: r.get("ts") or ""):
        th = ev.get("text_hash") or ev.get("analysis_id") or ""
        if th in seen:
            continue
        seen.add(th)
        v = ev.get("compliance_overall_pct")
        if isinstance(v, (int, float)):
            pcts.append(float(v))
            if float(v) < 70:
                risk_n += 1
    if pcts:
        avg_pct = round(sum(pcts) / len(pcts), 1)
        risk_rate = round(100.0 * risk_n / len(pcts), 1)

    compliance_hist = _compliance_buckets(kz_events)

    rating_hist = charts.get("rating_histogram") or {}
    rating_chart = [{"rating": int(k), "count": int(v)} for k, v in sorted(rating_hist.items()) if int(v) > 0]

    activity = charts.get("activity_by_day") or []
    if len(activity) > 14:
        activity = activity[-14:]

    rubric_chart = (charts.get("rubric_kz_runs") or [])[:8]

    readiness_items = ml.get("items") or []

    block_overrides = [
        {"block_key": row.get("block_key") or "?", "label": row.get("block_key") or "?", "count": int(row.get("count") or 0)}
        for row in (stats.get("block_overrides_top") or [])[:8]
        if int(row.get("count") or 0) > 0
    ]

    pm = protocol_match or {}
    total_kz_pm = int(pm.get("total_kz_runs") or kz_total or 0)
    with_matched = int(pm.get("kz_with_matched_protocol") or 0)
    kz_protocol_match = []
    if total_kz_pm > 0:
        kz_protocol_match = [
            {"label": "протокол сопоставлен", "count": with_matched},
            {"label": "без сопоставления", "count": max(0, total_kz_pm - with_matched)},
        ]

    out: dict[str, Any] = {
        "generated_at": _utc_now(),
        "live": has_real,
        "period_label": "Пилот Protocol · обезличенные агрегаты" if has_real else "Корпус и инфраструктура (ожидаем КЗ)",
        "version": version,
        "rag_ready": rag_ready,
        "protocols_in_corpus": (corpus or {}).get("protocols_in_index"),
        "chunks_loaded": (corpus or {}).get("chunks_loaded"),
        "reviews_total": reviews,
        "unique_kz": unique_kz,
        "kz_runs_total": kz_total,
        "avg_compliance_pct": avg_pct,
        "risk_zone_rate_pct": risk_rate,
        "readiness_overall_pct": summary.get("readiness_overall_pct"),
        "protocol_hit_at_3_pct": protocol_match.get("protocol_hit_at_3_pct"),
        "retrieval_fix_count": pool.get("retrieval_fixes") or 0,
        "priority_cases": summary.get("priority_cases") or 0,
        "charts": {
            "activity_by_day": activity,
            "rating_histogram": rating_chart,
            "rubric_kz_runs": rubric_chart,
            "compliance_buckets": compliance_hist,
            "readiness_items": [
                {
                    "label": it.get("label") or "",
                    "current": it.get("current"),
                    "target": it.get("target"),
                    "pct": it.get("pct"),
                }
                for it in readiness_items[:6]
            ],
            "tags_top": (charts.get("tags_top") or [])[:6],
            "block_overrides_top": block_overrides,
            "kz_protocol_match": kz_protocol_match,
        },
        "engine_releases_count": len(stats.get("engine_releases") or []),
        "verdict_breakdown": stats.get("verdict_breakdown") or {},
        "note_ru": (
            "Данные обновляются при каждом запросе; персональные данные и тексты КЗ не показываются."
            if has_real
            else "Разметка КЗ началась - метрики появятся после первых прогонов в кабинете методиста."
        ),
    }
    return out
