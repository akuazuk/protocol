"""Canonical lab identity and result lifecycle for MO review.

The contract is shadow-only. It gates findings conservatively and never promotes
lab signals into the primary score.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from typing import Any

from clinical_knowledge.lab_canons import lab_panels, text_hits_panel, type_hits_panel

ENGINE = "mo_lab_result_assessment_v1"
NONFINAL_STATES = frozenset(
    {"ordered", "collected", "in_progress", "pending", "processing", "cancelled"}
)
INTERPRETATION_MARKERS = (
    "повышен",
    "снижен",
    "выше нормы",
    "ниже нормы",
    "отклон",
    "патолог",
    "аномал",
)


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cutoff_datetime(cutoff_at: str, visit_date: str) -> datetime | None:
    parsed = _parse_datetime(cutoff_at)
    if parsed is not None:
        return parsed
    try:
        day = date.fromisoformat(str(visit_date or "")[:10])
    except ValueError:
        return None
    return datetime.combine(day, time.max, tzinfo=timezone.utc)


def _panel_for(type_name: str, analyte: str) -> dict[str, Any] | None:
    for panel in lab_panels():
        if type_hits_panel(analyte, panel):
            return panel
    for panel in lab_panels():
        if type_hits_panel(type_name, panel):
            return panel
    return None


def _availability(
    *,
    test_date: str,
    available_at: str,
    cutoff: datetime | None,
) -> str:
    if cutoff is None:
        return "unknown_cutoff"
    available = _parse_datetime(available_at)
    if available is not None:
        return "provably_available" if available <= cutoff else "post_cutoff"
    day = str(test_date or "")[:10]
    cutoff_day = cutoff.date().isoformat()
    if not day:
        return "unknown_date"
    if day < cutoff_day:
        return "provably_available"
    if day == cutoff_day:
        return "unknown_same_day"
    return "post_cutoff"


def _text(case: Mapping[str, Any], *keys: str) -> str:
    return "\n".join(str(case.get(key) or "") for key in keys if case.get(key))


def build_lab_result_assessment(
    bundle: Mapping[str, Any] | None,
    case: Mapping[str, Any] | None,
    *,
    cutoff_at: str = "",
) -> dict[str, Any]:
    """Return per-result identity/lifecycle without changing primary findings."""

    source = bundle if isinstance(bundle, Mapping) else {}
    enforcement_enabled = source.get("lifecycle_schema_available") is True
    current = case if isinstance(case, Mapping) else {}
    visit_date = str(
        current.get("visit_date")
        or current.get("date")
        or (source.get("window") or {}).get("visit_date")
        or ""
    )[:10]
    cutoff = _cutoff_datetime(cutoff_at, visit_date)
    mention_text = _text(
        current,
        "exam_data",
        "clinical_diagnosis",
        "diagnosis_main_text",
        "exam_recommendations",
        "treatment_recommendations",
    )
    plan_text = _text(
        current,
        "exam_recommendations",
        "treatment_recommendations",
        "dispensary_info",
        "return_date",
    )
    interpretation_blob = mention_text.lower().replace("ё", "е")
    try:
        from clinical_knowledge.patient_age import resolve_patient_age

        age_years = resolve_patient_age(dict(current)).get("age_years")
    except Exception:  # noqa: BLE001
        age_years = None

    try:
        from clinical_knowledge.lab_abnormal_findings import _match_range, _parse_number
    except ImportError:
        _match_range = None
        _parse_number = None

    results: list[dict[str, Any]] = []
    for day in source.get("days") or []:
        if not isinstance(day, Mapping):
            continue
        test_date = str(day.get("test_date") or "")[:10]
        for lab_type in day.get("types") or []:
            if not isinstance(lab_type, Mapping):
                continue
            type_name = str(lab_type.get("type_name") or "")
            for indicator in lab_type.get("indicators") or []:
                if not isinstance(indicator, Mapping):
                    continue
                analyte = str(indicator.get("name") or "")
                panel = _panel_for(type_name, analyte)
                available_at = str(
                    indicator.get("available_at") or lab_type.get("available_at") or ""
                )
                availability = _availability(
                    test_date=test_date,
                    available_at=available_at,
                    cutoff=cutoff,
                )
                status = str(
                    indicator.get("result_status")
                    or lab_type.get("result_status")
                    or ""
                ).strip().lower()
                has_value = bool(str(indicator.get("value") or "").strip())
                present = bool(has_value and status not in NONFINAL_STATES)
                ref = (
                    _match_range(analyte, str(indicator.get("unit") or ""))
                    if _match_range is not None and has_value
                    else None
                )
                if (
                    ref is not None
                    and ref.get("population") == "adult"
                    and (age_years is None or age_years < 18)
                ):
                    ref = None
                value = (
                    _parse_number(indicator.get("value"))
                    if _parse_number is not None and has_value
                    else None
                )
                abnormal: bool | None = None
                if ref is not None and value is not None:
                    abnormal = not (float(ref["low"]) <= value <= float(ref["high"]))
                mentioned = bool(
                    panel and text_hits_panel(mention_text, panel)
                ) or bool(analyte and analyte.lower() in mention_text.lower())
                interpreted: bool | None
                if (
                    not present
                    or availability != "provably_available"
                    or ref is None
                ):
                    interpreted = None
                else:
                    interpreted = bool(
                        mentioned
                        and any(marker in interpretation_blob for marker in INTERPRETATION_MARKERS)
                    )
                acted: bool | None = (
                    bool(panel and text_hits_panel(plan_text, panel))
                    if present and availability == "provably_available"
                    else None
                )
                results.append(
                    {
                        "identity": {
                            "test_id": lab_type.get("test_id"),
                            "order_ref": lab_type.get("order_ref") or None,
                            "panel_id": (panel or {}).get("id"),
                            "analyte": analyte or None,
                            "specimen": lab_type.get("specimen") or None,
                            "method": lab_type.get("method") or None,
                        },
                        "test_date": test_date or None,
                        "available_at": available_at or None,
                        "result_status": status or ("reported_legacy" if has_value else "unknown"),
                        "availability": availability,
                        "result_present": present,
                        "result_mentioned": mentioned,
                        "result_interpreted": interpreted,
                        "result_acted": acted,
                        "reference_available": ref is not None,
                        "is_abnormal": abnormal,
                        "actionable_for_review": bool(
                            present
                            and availability == "provably_available"
                            and abnormal is True
                        ),
                    }
                )

    reason = str(source.get("reason") or "")
    status = "completed" if results else (
        "empty" if reason == "empty" else "unavailable"
    )
    return {
        "contract_version": 1,
        "engine": ENGINE,
        "status": status,
        "reason": reason or None,
        "cutoff_at": cutoff.isoformat() if cutoff else None,
        "results": results,
        "summary": {
            "results_n": len(results),
            "provably_available_n": sum(
                row["availability"] == "provably_available" for row in results
            ),
            "unknown_availability_n": sum(
                str(row["availability"]).startswith("unknown") for row in results
            ),
            "post_cutoff_n": sum(row["availability"] == "post_cutoff" for row in results),
            "actionable_n": sum(bool(row["actionable_for_review"]) for row in results),
        },
        "enforcement_enabled": enforcement_enabled,
        "primary": False,
        "shadow": True,
    }


def actionable_panel_ids(assessment: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(assessment, Mapping):
        return set()
    return {
        str((row.get("identity") or {}).get("panel_id"))
        for row in assessment.get("results") or []
        if isinstance(row, Mapping)
        and row.get("actionable_for_review")
        and (row.get("identity") or {}).get("panel_id")
    }

