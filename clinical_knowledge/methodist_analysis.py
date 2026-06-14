"""Снимок прогона анализа КЗ для triage и очереди методиста."""
from __future__ import annotations

from typing import Any


def get_methodist_analysis(analysis_id: str) -> dict[str, Any] | None:
    """Возвращает сохранённый снимок api_result + метаданные (без пересчёта)."""
    from clinical_knowledge.feedback_store import load_analysis_snapshot, load_secure_kz_text

    aid = (analysis_id or "").strip()
    if not aid:
        return None
    snap = load_analysis_snapshot(aid)
    if not snap:
        return None
    text_hash = str(snap.get("text_hash") or "")
    full_text = load_secure_kz_text(text_hash) or ""
    api_result = snap.get("api_result")
    if not isinstance(api_result, dict):
        api_result = {}
    out: dict[str, Any] = {
        "analysis_id": snap.get("analysis_id") or aid,
        "text_hash": text_hash,
        "tier": snap.get("tier"),
        "saved_at": snap.get("saved_at"),
        "text_excerpt": snap.get("text_excerpt"),
        "has_full_text": bool(full_text.strip()),
        "api_result": api_result,
    }
    if full_text.strip():
        out["full_text"] = full_text
    return out
