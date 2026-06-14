"""Обезличенная телеметрия POST /api/assist (поиск протоколов)."""
from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _top_confidence_pct(proto_list: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for pr in proto_list:
        if not isinstance(pr, dict):
            continue
        sc = pr.get("confidence_score")
        if sc is None:
            continue
        try:
            v = float(sc)
            if v <= 1.0:
                v *= 100.0
            if best is None or v > best:
                best = v
        except (TypeError, ValueError):
            continue
    return round(best, 1) if best is not None else None


def log_protocol_search(
    *,
    query: str,
    retrieved: list[dict[str, Any]],
    proto_list: list[dict[str, Any]],
    icd_codes: list[str] | None,
    user_slugs: list[str] | None,
    audience_inferred: str | None,
) -> None:
    """Append-only событие без текста запроса (только длина и короткий hash)."""
    q = (query or "").strip()
    if not q:
        return
    try:
        from clinical_knowledge.feedback_store import append_feedback_event

        append_feedback_event(
            {
                "event_type": "protocol_search",
                "query_len": len(q),
                "query_hash": hashlib.sha256(q.encode("utf-8")).hexdigest()[:16],
                "has_icd": bool(icd_codes),
                "n_retrieval": len(retrieved or []),
                "n_protocols": len(proto_list or []),
                "top_confidence_pct": _top_confidence_pct(proto_list or []),
                "audience": (audience_inferred or "").strip() or None,
                "rubric_filters": len(user_slugs or []),
            }
        )
    except Exception as exc:
        _log.debug("protocol_search telemetry skipped: %s", exc)


def iter_protocol_search_events(feedback_dir: Path | None = None) -> list[dict[str, Any]]:
    from clinical_knowledge.feedback_store import feedback_dir as resolve_feedback_dir
    from clinical_knowledge.methodist_stats import iter_feedback_events

    fb = feedback_dir or resolve_feedback_dir()
    return [e for e in iter_feedback_events(fb) if e.get("event_type") == "protocol_search"]


def aggregate_protocol_search(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "total_searches": 0,
            "unique_queries": 0,
            "with_icd_pct": None,
            "avg_top_confidence_pct": None,
            "avg_protocols_returned": None,
        }

    unique_hashes: set[str] = set()
    icd_n = 0
    confs: list[float] = []
    protos: list[int] = []
    by_day: Counter[str] = Counter()

    for ev in events:
        qh = str(ev.get("query_hash") or "")
        if qh:
            unique_hashes.add(qh)
        if ev.get("has_icd"):
            icd_n += 1
        tc = ev.get("top_confidence_pct")
        if isinstance(tc, (int, float)):
            confs.append(float(tc))
        np = ev.get("n_protocols")
        if isinstance(np, int):
            protos.append(np)
        ts = str(ev.get("ts") or "")
        if len(ts) >= 10:
            by_day[ts[:10]] += 1

    total = len(events)
    activity = [{"date": d, "count": c} for d, c in sorted(by_day.items())]
    if len(activity) > 14:
        activity = activity[-14:]

    conf_buckets = [
        {"label": "≥85%", "count": sum(1 for c in confs if c >= 85)},
        {"label": "70–84%", "count": sum(1 for c in confs if 70 <= c < 85)},
        {"label": "<70%", "count": sum(1 for c in confs if c < 70)},
    ]

    return {
        "total_searches": total,
        "unique_queries": len(unique_hashes),
        "with_icd_pct": round(100.0 * icd_n / total, 1) if total else None,
        "avg_top_confidence_pct": round(sum(confs) / len(confs), 1) if confs else None,
        "avg_protocols_returned": round(sum(protos) / len(protos), 1) if protos else None,
        "activity_by_day": activity,
        "confidence_buckets": [b for b in conf_buckets if b["count"]],
    }
