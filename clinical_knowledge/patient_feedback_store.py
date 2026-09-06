"""Append-only telemetry B2C (без ПДн): hash, scores, quality flags."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from .feedback_store import feedback_dir
from .jsonl_io import append_line

_log = logging.getLogger(__name__)

PATIENT_REVIEW_LOG = "patient_review.jsonl"
PATIENT_UI_LOG = "patient_ui.jsonl"
PATIENT_NIGHTLY_LOG = "patient_nightly.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def text_hash(text: str) -> str:
    raw = (text or "").strip().encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:32]


def _append_jsonl(filename: str, row: dict[str, Any]) -> None:
    if os.environ.get("PATIENT_TELEMETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    append_line(feedback_dir() / filename, row)


def compute_quality_flags(
    *,
    kz_text: str,
    report: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    blob = str(report).lower()
    kz_low = (kz_text or "").lower()
    exams = report.get("extracted_exams") or []
    exam_types = {str(e.get("exam_type") or "") for e in exams if isinstance(e, dict)}
    has_mri_exam = "MRI" in exam_types
    if "мрт" in blob and "мрт" not in kz_low and not has_mri_exam:
        flags.append("no_mri_in_source_but_in_summary")
    imaging = [e for e in exams if isinstance(e, dict) and e.get("category") == "imaging"]
    understood = " ".join(
        str(it.get("value_ru") or "")
        for it in (report.get("understood_from_document") or [])
        if isinstance(it, dict)
    ).lower()
    if imaging and ("кт" in understood or "узи" in understood):
        for ex in imaging:
            label = str(ex.get("label_ru") or "").lower()
            if label and label not in kz_low:
                flags.append("false_imaging_in_understood")
                break
    if report.get("protocol_confidence_bucket") == "low" and report.get("matched_protocols_count", 0) > 0:
        flags.append("low_protocol_confidence")
    ctx = report.get("patient_context") or {}
    if not ctx.get("specialty") and (report.get("protocol_confidence") or 0) > 0.55:
        flags.append("specialty_unknown")
    qs = report.get("questions_for_doctor") or []
    if len(qs) >= 6 and len({q[:40].lower() for q in qs if isinstance(q, str)}) < len(qs) - 1:
        flags.append("duplicate_questions")
    return flags


def record_patient_review_snapshot(
    *,
    kz_text: str,
    report: dict[str, Any],
    build_version: str,
    latency_ms: int | None = None,
    has_lab_upload: bool = False,
) -> str:
    """Сохранить обезличенный снимок прогона B2C. Возвращает text_hash."""
    th = text_hash(kz_text)
    ctx = report.get("patient_context") if isinstance(report.get("patient_context"), dict) else {}
    exams = report.get("extracted_exams") or []
    meds = report.get("extracted_medications") or []
    flags = compute_quality_flags(kz_text=kz_text, report=report)
    scores = report.get("scores") or {}
    row: dict[str, Any] = {
        "event_type": "patient_review",
        "ts": _utc_now(),
        "text_hash": th,
        "build_version": build_version,
        "latency_ms": latency_ms,
        "context": {
            "specialty_inferred": ctx.get("specialty"),
            "icd10": ctx.get("icd10_codes") or [],
            "age_group": ctx.get("age_group"),
            "has_lab_upload": bool(has_lab_upload),
        },
        "extracted": {
            "exam_types": [e.get("exam_type") for e in exams if isinstance(e, dict)][:8],
            "med_count": len(meds),
            "imaging_count": sum(1 for e in exams if isinstance(e, dict) and e.get("category") == "imaging"),
            "lab_count": sum(1 for e in exams if isinstance(e, dict) and e.get("category") == "lab"),
        },
        "scores": {
            "document_completeness": (scores.get("document_completeness") or {}).get("pct"),
            "patient_clarity": (scores.get("patient_clarity") or {}).get("pct"),
            "protocol_match_confidence": (scores.get("protocol_match_confidence") or {}).get("pct"),
            "protocol_confidence_bucket": report.get("protocol_confidence_bucket"),
        },
        "quality_flags": flags,
        "question_count": len(report.get("questions_for_doctor") or []),
    }
    links = report.get("protocol_links") or []
    if links and isinstance(links[0], dict):
        row["primary_protocol_path"] = links[0].get("path")
    _append_jsonl(PATIENT_REVIEW_LOG, row)

    if os.environ.get("PATIENT_STORE_REPORT_SNAPSHOT", "1").strip().lower() not in ("0", "false", "no", "off"):
        _store_report_snapshot(th, report, build_version, flags)
    return th


def _store_report_snapshot(
    th: str,
    report: dict[str, Any],
    build_version: str,
    flags: list[str],
) -> None:
    from .feedback_store import analyses_dir

    out_dir = analyses_dir() / "patient"
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        "text_hash": th,
        "build_version": build_version,
        "ts": _utc_now(),
        "quality_flags": flags,
        "plain_summary_ru": report.get("plain_summary_ru") or (report.get("top_summary") or {}).get("plain_summary_ru"),
        "clarification_points": report.get("clarification_points") or [],
        "extracted_exams": report.get("extracted_exams") or [],
        "extracted_medications": report.get("extracted_medications") or [],
        "questions_for_doctor": report.get("questions_for_doctor") or [],
        "patient_context": report.get("patient_context") or {},
    }
    (out_dir / f"{th.replace(':', '_')}.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_patient_ui_event(
    *,
    event: str,
    text_hash_value: str | None = None,
    meta: dict[str, Any] | None = None,
    build_version: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "event_type": "patient_ui",
        "ts": _utc_now(),
        "event": event,
        "text_hash": text_hash_value,
        "build_version": build_version,
        "meta": meta or {},
    }
    _append_jsonl(PATIENT_UI_LOG, row)
