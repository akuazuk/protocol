"""B2C проверка КЗ (tier P1): L1 + patient report."""
from __future__ import annotations

from typing import Any

from .consult_tiering import run_l1_structured_review
from .patient_report import build_patient_report, sanitize_patient_api_payload


def run_patient_review(
    *,
    text: str,
    consultation_id: str = "patient",
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
    lab_text: str | None = None,
) -> dict[str, Any]:
    """P1: structured + alignment без ЦИСЗ и без LLM."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Пустой текст заключения")

    from .patient_exams_enrich import enrich_lab_crosscheck_with_exams_block
    from .patient_lab_crosscheck import crosscheck_labs_with_kz
    from .patient_protocol_crosscheck import crosscheck_protocol_requirements

    l1 = run_l1_structured_review(
        text=raw,
        consultation_id=consultation_id,
        demographics_meta=demographics_meta,
        specialty_slug=specialty_slug,
        skip_alignment=False,
    )
    align = l1.get("alignment") if isinstance(l1.get("alignment"), dict) else {}
    cards = list(align.get("alignment_cards") or [])
    exams_card = next((c for c in cards if isinstance(c, dict) and c.get("block_id") == "exams"), None)

    lab_check = crosscheck_labs_with_kz(kz_text=raw, lab_text=lab_text or "")
    if (lab_text or "").strip():
        lab_check = enrich_lab_crosscheck_with_exams_block(lab_check, exams_card=exams_card)

    protocol_context = crosscheck_protocol_requirements(
        l1_result=l1,
        kz_text=raw,
        lab_text=lab_text or "",
    )
    patient_report = build_patient_report(
        l1,
        lab_crosscheck=lab_check if (lab_text or "").strip() else None,
        protocol_context=protocol_context,
    )
    payload: dict[str, Any] = {
        "ok": True,
        "review_tier": "P1",
        "patient_report": patient_report,
        "confidence_score": l1.get("confidence_score"),
        "matched_protocols_count": l1.get("matched_protocols_count"),
        "llm_used": False,
        "rag_used": False,
        "criteria_source": l1.get("criteria_source") or "deterministic_alignment",
    }
    return sanitize_patient_api_payload(payload)


def patient_demographics_from_form(
    *,
    age_years: str | int | None = None,
    sex: str | None = None,
) -> dict[str, Any] | None:
    meta: dict[str, Any] = {}
    if age_years is not None and str(age_years).strip():
        try:
            age = int(str(age_years).strip())
            if 0 < age < 130:
                meta["age_years"] = age
        except ValueError:
            pass
    sx = (sex or "").strip().lower()
    if sx in ("male", "m", "муж", "мужской"):
        meta["sex"] = "male"
    elif sx in ("female", "f", "жен", "женский"):
        meta["sex"] = "female"
    return meta or None
