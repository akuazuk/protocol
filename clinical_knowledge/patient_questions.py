"""Спокойные вопросы врачу с deny-list (B2C default: calm_respectful)."""
from __future__ import annotations

import re
from typing import Any

from .patient_question_tone import (
    apply_tone_to_questions,
    detect_question_intent,
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
        "Подскажите, пожалуйста, в какие сроки нужно выполнить МРТ шейного отдела позвоночника и головного мозга?"
    ),
    "follow_up_timing": (
        "Повторный осмотр невролога нужен после МРТ или через определённое количество дней лечения?"
    ),
    "treatment_after": "Как правильно понимать слово «после» в схеме лечения?",
    "treatment_duration": "Сколько дней принимать препараты?",
    "symptoms_worse": "Что делать, если головная боль сохранится или усилится?",
    "urgent_symptoms": "При каких симптомах нужно обращаться срочно?",
    "exams_plan": "Какие обследования уже выполнены и какие необходимо пройти далее?",
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


def _calm_question_for_intent(intent: str | None, raw: str, block_id: str) -> str:
    if intent and intent in _CALM_TEMPLATES:
        return _CALM_TEMPLATES[intent]
    low = raw.lower()
    if "мрт" in low:
        return _CALM_TEMPLATES["exams_mri_deadline"]
    if "после" in low and block_id == "treatment":
        return _CALM_TEMPLATES["treatment_after"]
    if "длительност" in low or "сколько дней" in low:
        return _CALM_TEMPLATES["treatment_duration"]
    if "контрол" in low or "повторн" in low:
        return _CALM_TEMPLATES["follow_up_timing"]
    if "головн" in low and ("сохран" in low or "усил" in low):
        return _CALM_TEMPLATES["symptoms_worse"]
    return ""


def build_calm_questions(
    structured: list[dict[str, Any]],
    *,
    kz_text: str = "",
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
        block_id = str(row.get("block_id") or "")
        block_name = str(row.get("category_ru") or "")
        intent = row.get("intent") or detect_question_intent(raw, block_id, kind="comment" if comment else "gap")

        if use_calm:
            calm = _calm_question_for_intent(intent, raw, block_id)
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
            )

        text = sanitize_question_text(text)
        if not text or is_forbidden_question(text):
            continue
        norm = text.lower()[:80]
        if norm in seen:
            continue
        seen.add(norm)
        item = dict(row)
        item["text"] = text if text.endswith("?") else text.rstrip(".") + "?"
        item["title"] = item["text"].split("?")[0][:60] + "?"
        item["tone"] = DEFAULT_CALM_TONE if use_calm else tid
        item["intent"] = intent
        out.append(item)

    if "мрт" in (kz_text or "").lower():
        extra = [
            ("exams_mri_deadline", _CALM_TEMPLATES["exams_mri_deadline"]),
            ("follow_up_timing", _CALM_TEMPLATES["follow_up_timing"]),
            ("treatment_after", _CALM_TEMPLATES["treatment_after"]),
            ("treatment_duration", _CALM_TEMPLATES["treatment_duration"]),
            ("symptoms_worse", _CALM_TEMPLATES["symptoms_worse"]),
            ("urgent_symptoms", _CALM_TEMPLATES["urgent_symptoms"]),
        ]
        for intent_key, qtext in extra:
            if qtext.lower()[:60] in seen:
                continue
            seen.add(qtext.lower()[:60])
            out.append(
                {
                    "id": f"q_calm_{len(out)+1}",
                    "text": qtext,
                    "title": qtext.split("?")[0][:60] + "?",
                    "severity": "medium",
                    "category_ru": "Контроль" if "осмотр" in qtext else "Лечение",
                    "block_id": "follow_up" if "осмотр" in qtext else "treatment",
                    "tone": DEFAULT_CALM_TONE,
                    "intent": intent_key,
                }
            )

    return out[:10]


def apply_safe_questions(
    structured: list[dict[str, Any]],
    *,
    kz_text: str = "",
    tone: str | None = None,
    safety_enabled: bool = True,
) -> list[dict[str, Any]]:
    if not safety_enabled:
        return apply_tone_to_questions(structured, tone)
    return build_calm_questions(structured, kz_text=kz_text, tone=tone)
