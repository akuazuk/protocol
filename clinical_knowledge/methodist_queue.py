"""Очередь active learning для кабинета методиста."""
from __future__ import annotations

from typing import Any

from clinical_knowledge.methodist_stats import iter_feedback_events, _rubric_label, _short_hash
from clinical_knowledge.feedback_store import feedback_dir


def build_methodist_queue(*, limit: int = 50) -> dict[str, Any]:
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
