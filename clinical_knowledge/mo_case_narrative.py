"""Опциональный LLM-нарратив разбора МО (не меняет зоны).

См. docs/plans/2026-08-09-mo-case-review-quality-parity-v1.md фаза E.
Default off: MO_CASE_NARRATIVE=0. Живой Gemini - только GCE.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .phi_for_llm import redact_mapping_for_llm

ENGINE = "mo_case_narrative_v1"


def case_narrative_enabled() -> bool:
    raw = (os.environ.get("MO_CASE_NARRATIVE") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def build_narrative_prompt(brief: dict[str, Any], clinical: dict[str, Any]) -> str:
    return (
        "Ты методист МО РБ. По machine-brief и слотам МО напиши короткий JSON без markdown:\n"
        '{"summary_ru":"1-2 предложения","clinical_gaps_ru":["..."],'
        '"doctor_feedback_ru":["..."],"confidence":0.0}\n'
        "Не меняй и не выдумывай баллы зон. Только клиника и формулировки врачу.\n"
        # brief несёт visit_id, patient_id и ФИО врача; модели для формулировок
        # врачу они не нужны, поэтому уходят псевдонимами и инициалами.
        f"BRIEF:\n{json.dumps(redact_mapping_for_llm(brief), ensure_ascii=False)[:4000]}\n"
        f"CLINICAL:\n{json.dumps(clinical, ensure_ascii=False)[:3000]}\n"
    )


def normalize_narrative(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"ok": False, "available": False, "engine": ENGINE, "reason": "empty"}
    gaps = raw.get("clinical_gaps_ru") if isinstance(raw.get("clinical_gaps_ru"), list) else []
    feedback = (
        raw.get("doctor_feedback_ru") if isinstance(raw.get("doctor_feedback_ru"), list) else []
    )
    try:
        conf = float(raw.get("confidence"))
    except Exception:  # noqa: BLE001
        conf = 0.0
    return {
        "ok": True,
        "available": True,
        "engine": ENGINE,
        "summary_ru": str(raw.get("summary_ru") or "")[:500],
        "clinical_gaps_ru": [str(x)[:240] for x in gaps[:6]],
        "doctor_feedback_ru": [str(x)[:240] for x in feedback[:6]],
        "confidence": max(0.0, min(1.0, conf)),
        "shadow": True,
    }


def generate_case_narrative(
    *,
    brief: dict[str, Any],
    clinical: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Вызов Gemini при включённом флаге. Без ключа/ошибки - available=false."""
    if not force and not case_narrative_enabled():
        return {
            "ok": True,
            "available": False,
            "engine": ENGINE,
            "reason": "MO_CASE_NARRATIVE выключен",
        }
    try:
        from clinical_knowledge.gemini_lite import (
            _extract_text,
            gemini_available,
            generate_lite_json_response,
            get_lite_gemini_model,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "available": False,
            "engine": ENGINE,
            "reason": f"gemini_unavailable: {exc}",
        }
    if not gemini_available():
        return {
            "ok": False,
            "available": False,
            "engine": ENGINE,
            "reason": "GOOGLE_API_KEY not set",
        }
    prompt = build_narrative_prompt(brief, clinical)
    try:
        model = get_lite_gemini_model()
        resp = generate_lite_json_response(model, prompt, timeout=90)
        text = _extract_text(resp)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "available": False,
            "engine": ENGINE,
            "reason": f"gemini_error: {exc}",
        }
    parsed = _extract_json(str(text or ""))
    if not parsed:
        return {
            "ok": False,
            "available": False,
            "engine": ENGINE,
            "reason": "json_parse_failed",
        }
    return normalize_narrative(parsed)
