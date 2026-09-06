"""Слой B: слоты прошлых визитов эпизода + shadow-кредит.

Не меняет official overall_pct. Полные тексты prior в API не отдаём - только
какие слоты были и дата. Глубокий прогон на диске может держать слоты у себя.
"""
from __future__ import annotations

from typing import Any, Mapping

from clinical_knowledge.mo_case_document import (
    CLINICAL_FIELDS,
    clinical_fields_from_row,
    load_case_source_row,
)
from clinical_knowledge.mo_history_continuity import (
    MODE_KNOWN_DOCTOR,
    MODE_KNOWN_SPECIALTY,
    evaluate_history_continuity,
    _same_episode,
    _stem,
)

ENGINE = "mo_history_deep_v1"
FINDING_CODE = "B_history_episode_credit"
SLOT_KEYS = [key for key, _label in CLINICAL_FIELDS]


def _visits_from_bundle(bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(bundle, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for shelf in ("same_doctor", "same_specialty"):
        for row in bundle.get(shelf) or []:
            if isinstance(row, Mapping):
                item = dict(row)
                item["_shelf"] = shelf
                out.append(item)
    return out


def load_prior_slots_for_visits(
    visits: list[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Прочитать клинические слоты prior из secure CSV. Без patient_id в ответе."""
    loaded: list[dict[str, Any]] = []
    for visit in visits[: max(0, int(limit))]:
        visit_id = str(visit.get("visit_id") or visit.get("mis_id") or "").strip()
        day = str(visit.get("visit_date") or "")[:10]
        if not visit_id or len(day) < 10:
            continue
        try:
            row = load_case_source_row(
                visit_id,
                visit_date=day,
                mis_id=str(visit.get("mis_id") or "") or None,
            )
        except Exception:  # noqa: BLE001
            row = None
        if not row:
            continue
        clinical = clinical_fields_from_row(row)
        if not clinical:
            continue
        loaded.append(
            {
                "visit_date": day,
                "visit_id": visit_id,
                "mis_id": str(visit.get("mis_id") or ""),
                "shelf": str(visit.get("_shelf") or ""),
                "diagnosis_code": str(visit.get("diagnosis_code") or ""),
                "present_slots": [key for key in SLOT_KEYS if clinical.get(key)],
                "clinical": clinical,
            }
        )
    return loaded


def pick_episode_prior(
    *,
    history_bundle: Mapping[str, Any] | None,
    current_code: str = "",
    current_text: str = "",
    limit: int = 3,
) -> dict[str, Any]:
    """Лучший prior того же эпизода + список слотов (даты, без сырого patient_id)."""
    continuity = evaluate_history_continuity(
        current_code=current_code,
        current_text=current_text,
        history_bundle=history_bundle,
    )
    visits = _visits_from_bundle(history_bundle)
    if continuity.get("mode") == MODE_KNOWN_DOCTOR:
        visits = [row for row in visits if row.get("_shelf") == "same_doctor"]
    elif continuity.get("mode") == MODE_KNOWN_SPECIALTY:
        visits = [row for row in visits if row.get("_shelf") == "same_specialty"]
    else:
        # An unrelated earlier encounter cannot supply credit for this episode.
        visits = []
    summary = (history_bundle or {}).get("summary") or {}
    stem = _stem(current_code or str(summary.get("current_code") or ""))
    visits = [
        row for row in visits
        if _same_episode(current_stem=stem, current_text=current_text, visit=row)
    ]
    visits.sort(key=lambda row: str(row.get("visit_date") or ""), reverse=True)
    slots = load_prior_slots_for_visits(visits, limit=limit)
    richest = max(
        slots,
        key=lambda item: (str(item.get("visit_date") or ""), len(item.get("present_slots") or [])),
        default=None,
    )
    public_slots = [
        {
            "visit_date": item.get("visit_date"),
            "shelf": item.get("shelf"),
            "diagnosis_code": item.get("diagnosis_code"),
            "present_slots": item.get("present_slots") or [],
        }
        for item in slots
    ]
    prior_clinical = dict(richest.get("clinical") or {}) if richest else None
    return {
        "engine": ENGINE,
        "continuity": continuity,
        "prior_selection": "matched_episode_then_recent_then_complete",
        "prior_n_loaded": len(slots),
        "prior_slots": public_slots,
        "prior_clinical": prior_clinical,
        "prior_visit_date": (richest or {}).get("visit_date") or "",
        "already_slots": (richest or {}).get("present_slots") or [],
    }


def shadow_history_credit_finding(deep: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(deep, Mapping):
        return None
    continuity = deep.get("continuity") if isinstance(deep.get("continuity"), Mapping) else {}
    if not continuity.get("known_episode"):
        return None
    slots = deep.get("already_slots") or []
    date = str(deep.get("prior_visit_date") or continuity.get("last_matched_date") or "")
    slot_ru = {
        "complaints": "жалобы",
        "anamnesis_doctor": "анамнез",
        "objective_status": "статус",
        "clinical_diagnosis": "диагноз",
        "exam_recommendations": "план обследования",
        "treatment_recommendations": "план лечения",
    }
    named = [slot_ru[key] for key in slots if key in slot_ru]
    detail = "Продолжение эпизода"
    if date:
        detail += f" (последний prior {date})"
    if named:
        detail += ". Уже было: " + ", ".join(named)
    detail += ". Сегодняшний осмотр и актуальный план всё равно обязательны. Официальный балл не меняем."
    return {
        "code": FINDING_CODE,
        "finding_code": FINDING_CODE,
        "axis": "patient_history",
        "severity": "P3",
        "title_ru": str(continuity.get("mode_ru") or "История эпизода"),
        "detail_ru": detail,
        "shadow": True,
        "source": ENGINE,
    }


def public_deep_for_ui(deep: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(deep, Mapping):
        return {"engine": ENGINE, "prior_n_loaded": 0, "prior_slots": []}
    return {
        "engine": ENGINE,
        "prior_n_loaded": int(deep.get("prior_n_loaded") or 0),
        "prior_slots": list(deep.get("prior_slots") or []),
        "prior_visit_date": str(deep.get("prior_visit_date") or ""),
        "already_slots": list(deep.get("already_slots") or []),
    }
