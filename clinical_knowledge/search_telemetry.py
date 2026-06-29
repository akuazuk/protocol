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
    retrieved: list[dict[str, Any]] | None = None,
    proto_list: list[dict[str, Any]] | None = None,
    icd_codes: list[str] | None = None,
    user_slugs: list[str] | None = None,
    audience_inferred: str | None = None,
    search_source: str | None = None,
    n_retrieval: int | None = None,
) -> None:
    """Append-only событие без текста запроса (только длина и короткий hash)."""
    q = (query or "").strip()
    if not q:
        return
    try:
        from clinical_knowledge.feedback_store import append_feedback_event

        event: dict[str, Any] = {
            "event_type": "protocol_search",
            "query_len": len(q),
            "query_hash": hashlib.sha256(q.encode("utf-8")).hexdigest()[:16],
            "has_icd": bool(icd_codes),
            "n_retrieval": n_retrieval if n_retrieval is not None else len(retrieved or []),
            "n_protocols": len(proto_list or []),
            "top_confidence_pct": _top_confidence_pct(proto_list or []),
            "audience": (audience_inferred or "").strip() or None,
            "rubric_filters": len(user_slugs or []),
        }
        if search_source:
            event["search_source"] = search_source
        append_feedback_event(event)
    except Exception as exc:
        _log.debug("protocol_search telemetry skipped: %s", exc)


def log_protocol_search_from_payload(
    *,
    query: str,
    payload: dict[str, Any],
    icd_codes: list[str] | None = None,
    user_slugs: list[str] | None = None,
    audience_inferred: str | None = None,
    search_source: str = "assist",
) -> None:
    """Телеметрия из ответа assist / hybrid / icd-fast (без дубля логики полей)."""
    llm = payload.get("llm_json") if isinstance(payload.get("llm_json"), dict) else {}
    proto_list = llm.get("protocols") if isinstance(llm.get("protocols"), list) else []
    icd_src = icd_codes
    if not icd_src:
        icd_block = payload.get("icd")
        if isinstance(icd_block, dict):
            icd_src = list(icd_block.get("codes_for_retrieval") or icd_block.get("detected") or [])
    n_retrieval = payload.get("retrieved_count")
    if not isinstance(n_retrieval, int):
        n_retrieval = len(proto_list)
    log_protocol_search(
        query=query,
        proto_list=proto_list,
        icd_codes=icd_src,
        user_slugs=user_slugs,
        audience_inferred=audience_inferred,
        search_source=search_source,
        n_retrieval=int(n_retrieval) if isinstance(n_retrieval, int) else len(proto_list),
    )


_VALID_SEARCH_VERDICTS = ("fit", "miss")


def _basename(path: str | None) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.rsplit("/", 1)[-1] if p else ""


def log_search_feedback(
    *,
    query: str,
    verdict: str,
    rejected_path: str | None = None,
    chosen_path: str | None = None,
    top_paths: list[str] | None = None,
    icd_codes: list[str] | None = None,
    source: str = "doctor_search",
) -> str:
    """Лёгкий обезличенный фидбэк врача по подбору протокола (без текста запроса)."""
    v = (verdict or "").strip().lower()
    if v not in _VALID_SEARCH_VERDICTS:
        raise ValueError("verdict должен быть fit или miss")
    q = (query or "").strip()
    from clinical_knowledge.feedback_store import append_feedback_event

    event: dict[str, Any] = {
        "event_type": "search_feedback",
        "verdict": v,
        "source": (source or "doctor_search").strip()[:40],
        "query_len": len(q),
        "query_hash": hashlib.sha256(q.encode("utf-8")).hexdigest()[:16] if q else "",
        "rejected_basename": _basename(rejected_path),
        "chosen_basename": _basename(chosen_path),
        "top_basenames": [_basename(p) for p in (top_paths or [])[:5] if p],
        "has_icd": bool(icd_codes),
    }
    return append_feedback_event(event)


def iter_search_feedback_events(feedback_dir: Path | None = None) -> list[dict[str, Any]]:
    from clinical_knowledge.feedback_store import feedback_dir as resolve_feedback_dir
    from clinical_knowledge.methodist_stats import iter_feedback_events

    fb = feedback_dir or resolve_feedback_dir()
    return [e for e in iter_feedback_events(fb) if e.get("event_type") == "search_feedback"]


def aggregate_search_feedback(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    fit = sum(1 for e in events if str(e.get("verdict") or "").lower() == "fit")
    miss = total - fit
    miss_basenames: Counter[str] = Counter()
    recent: list[dict[str, Any]] = []
    for ev in events:
        if str(ev.get("verdict") or "").lower() == "miss":
            bn = str(ev.get("rejected_basename") or "")
            if bn:
                miss_basenames[bn] += 1
        recent.append(
            {
                "ts": ev.get("ts"),
                "verdict": ev.get("verdict"),
                "rejected_basename": ev.get("rejected_basename") or "",
                "chosen_basename": ev.get("chosen_basename") or "",
                "has_icd": bool(ev.get("has_icd")),
            }
        )
    recent = sorted(recent, key=lambda r: str(r.get("ts") or ""), reverse=True)[:50]
    return {
        "total": total,
        "fit": fit,
        "miss": miss,
        "fit_pct": round(100.0 * fit / total, 1) if total else None,
        "top_miss_protocols": [
            {"basename": b, "count": c} for b, c in miss_basenames.most_common(10)
        ],
        "recent": recent,
        "readiness": {
            "have": total,
            "target_golden": 20,
            "target_lora": 50,
        },
    }


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
            "activity_by_day": [],
            "confidence_buckets": [],
            "icd_usage": [],
            "rubric_filter_usage": [],
            "protocols_returned_buckets": [],
            "audience_breakdown": [],
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
        {"label": "70-84%", "count": sum(1 for c in confs if 70 <= c < 85)},
        {"label": "<70%", "count": sum(1 for c in confs if c < 70)},
    ]

    icd_usage: list[dict[str, Any]] = []
    rubric_filter_usage: list[dict[str, Any]] = []
    protocols_buckets: list[dict[str, Any]] = []
    audience_labels = {
        "adult": "взрослые",
        "child": "детские",
        "pediatric": "детские",
        "mixed": "смешанные",
    }
    audience_counts: Counter[str] = Counter()
    if total:
        icd_usage = [
            {"label": "с кодом МКБ", "count": icd_n},
            {"label": "без МКБ", "count": max(0, total - icd_n)},
        ]
        with_rubric = sum(1 for ev in events if int(ev.get("rubric_filters") or 0) > 0)
        rubric_filter_usage = [
            {"label": "с фильтром рубрик", "count": with_rubric},
            {"label": "без фильтра", "count": max(0, total - with_rubric)},
        ]
        pb = {"0": 0, "1-2": 0, "3-5": 0, "6+": 0}
        for ev in events:
            np = ev.get("n_protocols")
            if not isinstance(np, int):
                continue
            if np <= 0:
                pb["0"] += 1
            elif np <= 2:
                pb["1-2"] += 1
            elif np <= 5:
                pb["3-5"] += 1
            else:
                pb["6+"] += 1
        protocols_buckets = [{"label": k, "count": v} for k, v in pb.items() if v > 0]
        for ev in events:
            aud = str(ev.get("audience") or "").strip().lower() or "не указано"
            audience_counts[audience_labels.get(aud, aud if aud != "не указано" else "не указано")] += 1
    audience_breakdown = [{"label": k, "count": v} for k, v in audience_counts.most_common()]

    return {
        "total_searches": total,
        "unique_queries": len(unique_hashes),
        "with_icd_pct": round(100.0 * icd_n / total, 1) if total else None,
        "avg_top_confidence_pct": round(sum(confs) / len(confs), 1) if confs else None,
        "avg_protocols_returned": round(sum(protos) / len(protos), 1) if protos else None,
        "activity_by_day": activity,
        "confidence_buckets": [b for b in conf_buckets if b["count"]],
        "icd_usage": icd_usage,
        "rubric_filter_usage": rubric_filter_usage,
        "protocols_returned_buckets": protocols_buckets,
        "audience_breakdown": audience_breakdown,
    }
