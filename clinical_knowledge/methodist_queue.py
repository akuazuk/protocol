"""Очередь active learning для кабинета методиста."""
from __future__ import annotations

import hashlib
from typing import Any

from clinical_knowledge.methodist_stats import iter_feedback_events, _rubric_label, _short_hash
from clinical_knowledge.feedback_store import feedback_dir

_SEARCH_BAD_VERDICTS = frozenset({"partially_wrong", "wrong"})


def build_methodist_queue(*, limit: int = 50, domain: str | None = None) -> dict[str, Any]:
    if domain == "search":
        return build_search_methodist_queue(limit=limit)
    return build_kz_methodist_queue(limit=limit)


def build_kz_methodist_queue(*, limit: int = 50) -> dict[str, Any]:
    events = iter_feedback_events(feedback_dir())
    kz_by_id: dict[str, dict[str, Any]] = {}
    kz_by_hash: dict[str, dict[str, Any]] = {}
    reviews_by_aid: dict[str, dict[str, Any]] = {}
    reviews_by_hash: dict[str, dict[str, Any]] = {}

    for ev in events:
        et = ev.get("event_type")
        if et == "kz_analysis":
            aid = str(ev.get("analysis_id") or "")
            if aid:
                kz_by_id[aid] = ev
            th = str(ev.get("text_hash") or "")
            if th:
                kz_by_hash[th] = ev
        elif et == "analysis_review":
            aid = str(ev.get("analysis_id") or "")
            if aid:
                reviews_by_aid[aid] = ev
            th = str(ev.get("text_hash") or "")
            if th:
                reviews_by_hash[th] = ev

    priority: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    pending_aids: set[str] = set()

    seen_hash_priority: set[str] = set()
    for rev in sorted(reviews_by_aid.values(), key=lambda e: e.get("ts") or "", reverse=True):
        try:
            rating = int(rev.get("rating"))
        except (TypeError, ValueError):
            continue
        if rating > 2:
            continue
        th = str(rev.get("text_hash") or "")
        if th in seen_hash_priority:
            continue
        seen_hash_priority.add(th)
        kz = kz_by_id.get(str(rev.get("analysis_id") or "")) or kz_by_hash.get(th) or {}
        priority.append(_queue_row(kz, rev, reason="priority_rating_le_2"))

    reviewed_aids = set(reviews_by_aid.keys())
    for aid, kz in sorted(kz_by_id.items(), key=lambda x: x[1].get("ts") or "", reverse=True):
        if aid in reviewed_aids:
            continue
        th = str(kz.get("text_hash") or "")
        if any(r.get("text_hash") == th for r in priority):
            continue
        pending.append(_queue_row(kz, None, reason="pending_review"))
        pending_aids.add(aid)

    for aid, kz in kz_by_id.items():
        if aid in reviewed_aids or aid in pending_aids:
            continue
        failed = kz.get("failed_rule_ids") or []
        rules_pct = kz.get("rules_compliance_pct")
        if len(failed) >= 3 or rules_pct == 0.0:
            suspicious.append(_queue_row(kz, None, reason="suspicious_rules"))

    def _trim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[: max(1, limit // 3)]

    return {
        "domain": "kz",
        "priority": _trim(priority),
        "pending": _trim(pending),
        "suspicious": _trim(suspicious),
        "counts": {
            "priority": len(priority),
            "pending": len(pending),
            "suspicious": len(suspicious),
            "total_kz": len(kz_by_id),
            "total_reviews": len(reviews_by_aid),
        },
    }


def build_search_methodist_queue(*, limit: int = 50) -> dict[str, Any]:
    """Очередь разметки поиска: AI-verdict, retrieval_fix, низкая уверенность."""
    events = iter_feedback_events(feedback_dir())
    retrieval_fixes = [e for e in events if e.get("event_type") == "retrieval_fix"]
    search_reviews = [e for e in events if e.get("event_type") == "search_review"]
    protocol_searches = [e for e in events if e.get("event_type") == "protocol_search"]

    search_fixes = [
        f
        for f in retrieval_fixes
        if str(f.get("source") or "") == "protocol_search_ui"
        or f.get("review_source") in ("ai_assisted", "ai_assisted_edited", "manual")
    ]

    priority: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _key(ev: dict[str, Any]) -> str:
        qh = str(ev.get("query_hash") or "").strip()
        if qh:
            return "qh:" + qh
        q = str(ev.get("query") or "").strip()
        if q:
            return "q:" + hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
        return "e:" + str(ev.get("event_id") or "")

    def _add(bucket: list[dict[str, Any]], ev: dict[str, Any], *, reason: str) -> None:
        k = _key(ev)
        if k in seen_keys:
            return
        seen_keys.add(k)
        bucket.append(_search_queue_row(ev, reason=reason))

    combined = sorted(
        search_fixes + search_reviews,
        key=lambda e: e.get("ts") or "",
        reverse=True,
    )
    for ev in combined:
        ai = ev.get("ai_review") if isinstance(ev.get("ai_review"), dict) else {}
        verdict = str(ev.get("ranking_verdict") or ai.get("ranking_verdict") or "").strip()
        tags = list(ev.get("tags") or ai.get("tags") or [])
        if verdict in _SEARCH_BAD_VERDICTS:
            _add(priority, ev, reason=f"ranking_verdict_{verdict}")
        elif ev.get("event_type") == "retrieval_fix" and str(ev.get("source") or "") == "protocol_search_ui":
            _add(priority, ev, reason="retrieval_fix_labeled")
        if "query_too_vague" in tags:
            _add(suspicious, ev, reason="query_too_vague")

    for ev in sorted(protocol_searches, key=lambda e: e.get("ts") or "", reverse=True):
        tc = ev.get("top_confidence_pct")
        if isinstance(tc, (int, float)) and float(tc) < 70:
            _add(pending, ev, reason="low_top_confidence")
        elif ev.get("n_protocols") == 0:
            _add(pending, ev, reason="empty_protocol_list")

    def _trim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[: max(1, limit // 3)]

    return {
        "domain": "search",
        "priority": _trim(priority),
        "pending": _trim(pending),
        "suspicious": _trim(suspicious),
        "counts": {
            "priority": len(priority),
            "pending": len(pending),
            "suspicious": len(suspicious),
            "search_retrieval_fix": len(search_fixes),
            "search_reviews": len(search_reviews),
            "protocol_search_events": len(protocol_searches),
        },
    }


def _search_queue_row(ev: dict[str, Any], *, reason: str) -> dict[str, Any]:
    ai = ev.get("ai_review") if isinstance(ev.get("ai_review"), dict) else {}
    qh_raw = str(ev.get("query_hash") or "").strip()
    if not qh_raw:
        q = str(ev.get("query") or "").strip()
        if q:
            qh_raw = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
    top_paths = list(ev.get("retrieval_top_paths") or [])[:5]
    return {
        "event_id": ev.get("event_id"),
        "event_type": ev.get("event_type"),
        "query_hash_short": _short_hash(qh_raw) if qh_raw else None,
        "verdict": ev.get("ranking_verdict") or ai.get("ranking_verdict"),
        "ranking_rating": ev.get("ranking_rating") or ai.get("ranking_rating"),
        "top_paths": top_paths,
        "tags": list(ev.get("tags") or ai.get("tags") or [])[:6],
        "ts": ev.get("ts"),
        "reason": reason,
    }


def _queue_row(kz: dict[str, Any], rev: dict[str, Any] | None, *, reason: str) -> dict[str, Any]:
    rubric = str(kz.get("rubric") or "")
    return {
        "analysis_id": kz.get("analysis_id") or (rev or {}).get("analysis_id"),
        "text_hash_short": _short_hash(str(kz.get("text_hash") or (rev or {}).get("text_hash") or "")),
        "rubric": rubric,
        "rubric_label": _rubric_label(rubric),
        "tier": kz.get("tier"),
        "overall_pct": kz.get("compliance_overall_pct"),
        "rules_pct": kz.get("rules_compliance_pct"),
        "failed_rule_ids": (kz.get("failed_rule_ids") or [])[:5],
        "ts": kz.get("ts") or (rev or {}).get("ts"),
        "reason": reason,
        "rating": (rev or {}).get("rating"),
        "verdict": (rev or {}).get("verdict"),
        "tags": (rev or {}).get("tags") or [],
    }
