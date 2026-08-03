"""Спокойные вопросы врачу с deny-list (B2C default: calm_respectful)."""
from __future__ import annotations

import re
from typing import Any

from .patient_question_tone import (
    apply_tone_to_questions,
    detect_question_intent,
    is_playful_meta_template,
    normalize_question_tone,
    render_doctor_question,
)

DEFAULT_CALM_TONE = "calm_respectful"

FORBIDDEN_PATTERNS_RU: tuple[re.Pattern[str], ...] = (
    re.compile(r"анамнез\s+как\s+черновик", re.I),
    re.compile(r"осмотр\s+был,\s*а\s+половина\s+пропала", re.I),
    re.compile(r"по\s+протоколу\s+положено", re.I),
    re.compile(r"врач\s+ошиб", re.I),
    re.compile(r"не\s+принимайте", re.I),
    re.compile(r"назначен\s+неправильно", re.I),
    re.compile(r"уточните\s+у\s+врача", re.I),
)

_CALM_TEMPLATES: dict[str, str] = {
    "exams_mri_deadline": (
        "Подскажите, пожалуйста, в какие сроки нужно выполнить назначенное МРТ?"
    ),
    "follow_up_timing": (
        "Повторный осмотр нужен после обследований или через определённое количество дней лечения?"
    ),
    "treatment_after": "Как правильно понимать слово «после» в схеме лечения?",
    "treatment_duration": "Сколько дней принимать препараты?",
    "symptoms_worse": "Что делать, если симптомы сохранятся или усилятся?",
    "urgent_symptoms": "При каких симптомах нужно обращаться срочно?",
    "exams_plan": "Какие обследования уже выполнены и какие необходимо пройти далее?",
    "labs_plan": "Когда сдать назначенные анализы и как подготовиться?",
    "follow_up": "Когда следующий визит и что подготовить к приёму?",
    "treatment_dose": "Подскажите, пожалуйста, как правильно принимать назначенные препараты - дозу, время и длительность?",
}


def is_forbidden_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return any(p.search(t) for p in FORBIDDEN_PATTERNS_RU)


def sanitize_question_text(text: str) -> str:
    t = (text or "").strip()
    for pat in FORBIDDEN_PATTERNS_RU:
        if pat.search(t):
            return ""
    t = re.sub(r"по\s+протоколу\s+положено", "по стандарту лечения", t, flags=re.I)
    return t.strip()


def _has_mri(exams: list[dict[str, Any]] | None) -> bool:
    return any(e.get("exam_type") == "MRI" for e in (exams or []))


def _calm_question_for_intent(
    intent: str | None,
    raw: str,
    block_id: str,
    *,
    exams: list[dict[str, Any]] | None = None,
) -> str:
    low = raw.lower()
    if "мрт" in low and _has_mri(exams):
        return _CALM_TEMPLATES["exams_mri_deadline"]
    if intent == "exams_mri_deadline" and not _has_mri(exams):
        intent = "exams_plan"
    if intent and intent in _CALM_TEMPLATES:
        if intent == "exams_mri_deadline" and not _has_mri(exams):
            return _CALM_TEMPLATES["exams_plan"]
        return _CALM_TEMPLATES[intent]
    if "после" in low and block_id == "treatment":
        return _CALM_TEMPLATES["treatment_after"]
    if "длительност" in low or "сколько дней" in low:
        return _CALM_TEMPLATES["treatment_duration"]
    if "контрол" in low or "повторн" in low:
        return _CALM_TEMPLATES["follow_up_timing"]
    if "анализ" in low or block_id == "labs":
        return _CALM_TEMPLATES["labs_plan"]
    return ""


def build_calm_questions(
    structured: list[dict[str, Any]],
    *,
    kz_text: str = "",
    exams: list[dict[str, Any]] | None = None,
    tone: str | None = None,
) -> list[dict[str, Any]]:
    """Переформулировать вопросы в спокойном тоне с фильтром deny-list."""
    tid = normalize_question_tone(tone or "serious")
    use_calm = tid in ("serious", "calm_respectful", "official") or tone in (None, "", "calm_respectful")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    playful_used: set[str] = set()

    for i, row in enumerate(structured):
        if not isinstance(row, dict):
            continue
        gap = str(row.get("source_gap") or "").strip()
        comment = str(row.get("source_comment") or "").strip()
        raw = comment or gap
        preset = str(row.get("text") or "").strip()
        plain_ctx = str(row.get("plain_context") or "").strip()
        block_id = str(row.get("block_id") or "")
        block_name = str(row.get("category_ru") or "")
        intent = row.get("intent") or detect_question_intent(raw, block_id, kind="comment" if comment else "gap")

        if use_calm:
            calm = _calm_question_for_intent(intent, raw, block_id, exams=exams)
            if calm and calm.lower() not in seen:
                text = calm
            else:
                text, intent = render_doctor_question(
                    gap=gap,
                    comment=comment,
                    block_id=block_id,
                    block_name=block_name,
                    category_ru=block_name,
                    tone="serious",
                    intent=intent,
                    playful_slot=i,
                    playful_used=playful_used,
                    fallback_text=preset,
                    plain_context=plain_ctx,
                )
        else:
            text, intent = render_doctor_question(
                gap=gap,
                comment=comment,
                block_id=block_id,
                block_name=block_name,
                category_ru=block_name,
                tone=tid,
                intent=intent,
                playful_slot=i,
                playful_used=playful_used,
                fallback_text=preset,
                plain_context=plain_ctx,
            )
            if is_playful_meta_template(text) and preset:
                text = preset

        text = sanitize_question_text(text)
        if not text or is_forbidden_question(text):
            continue
        if "мрт" in text.lower() and not _has_mri(exams) and "мрт" not in (kz_text or "").lower():
            continue
        norm = text.lower()[:80]
        if norm in seen:
            continue
        seen.add(norm)
        item = dict(row)
        item["text"] = text if text.endswith("?") else text.rstrip(".") + "?"
        item["title"] = item["text"].split("?")[0].strip()[:60] + "?"
        item["tone"] = DEFAULT_CALM_TONE if use_calm else tid
        item["intent"] = intent
        out.append(item)

    return out[:10]


def apply_safe_questions(
    structured: list[dict[str, Any]],
    *,
    kz_text: str = "",
    exams: list[dict[str, Any]] | None = None,
    tone: str | None = None,
    safety_enabled: bool = True,
) -> list[dict[str, Any]]:
    if not safety_enabled:
        return apply_tone_to_questions(structured, tone)
    return build_calm_questions(structured, kz_text=kz_text, exams=exams, tone=tone)
