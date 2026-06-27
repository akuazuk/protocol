"""Privacy-safe аналитика B2C (без ПДн и текста КЗ)."""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("protocol.patient.analytics")

ALLOWED_EVENTS = frozenset(
    {
        "upload_start",
        "upload_done",
        "report_view",
        "share_tap",
        "print_tap",
        "checklist_item",
        "checklist_complete",
        "payment_start",
        "payment_done",
        "restore_report",
        "install_pwa_prompt",
        "reminder_set",
        "review_started",
        "review_completed",
        "result_viewed",
        "checklist_opened",
        "question_copied",
        "visit_sheet_downloaded",
        "share_clicked",
        "delete_data_clicked",
        "low_confidence_shown",
        "patient_result_aha",
        "message_copied",
        "visit_sheet_copied",
    }
)


def record_patient_event(
    *,
    event: str,
    clinic_id: str | None = None,
    tier_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = (event or "").strip().lower()
    if name not in ALLOWED_EVENTS:
        return {"ok": False, "error": "unknown_event"}
    safe_meta: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if k in (
            "latency_ms",
            "light",
            "block_count",
            "lab_count",
            "checked_count",
            "pct",
            "time_to_result_ms",
            "has_questions",
            "has_visit_sheet",
            "protocol_confidence_bucket",
            "document_quality_bucket",
            "tier",
            "upload_mismatch",
        ):
            if isinstance(v, (int, float, str, bool)) or v is None:
                safe_meta[k] = v
    payload = {
        "ts": int(time.time()),
        "event": name,
        "clinic_id": (clinic_id or "")[:32] or None,
        "tier_id": (tier_id or "")[:24] or None,
        "meta": safe_meta,
    }
    logger.info("patient_event %s", payload)
    return {"ok": True, "recorded": name}
