"""Тон формулировок «вопросов врачу» для B2C (уважительно, без конфликта с врачом)."""
from __future__ import annotations

import re
from typing import Any, Literal

QuestionTone = Literal["friendly", "serious", "official", "light"]

DEFAULT_QUESTION_TONE: QuestionTone = "friendly"

_TONE_ALIASES: dict[str, QuestionTone] = {
    "friendly": "friendly",
    "дружелюбно": "friendly",
    "warm": "friendly",
    "serious": "serious",
    "серьёзно": "serious",
    "серьезно": "serious",
    "official": "official",
    "официально": "official",
    "formal": "official",
    "light": "light",
    "лёгкий": "light",
    "легкий": "light",
    "humor": "light",
    "шуточно": "light",
    "юмор": "light",
}

CATEGORY_EMOJI: dict[str, str] = {
    "Жалобы": "🗣️",
    "Анамнез": "📋",
    "Осмотр": "🩺",
    "Диагноз": "🧬",
    "Обследования": "🔬",
    "Лечение": "💊",
    "Контроль": "📅",
    "Анализы": "🧪",
    "Протокол": "📑",
    "Документ": "📄",
    "Вопрос на приёме": "💬",
}

QUESTION_TONE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "friendly",
        "label_ru": "Дружелюбно",
        "description_ru": "Тепло и по-человечески - как на обычном приёме. Рекомендуем по умолчанию.",
        "emoji": "💬",
        "default": True,
        "doctor_safe": True,
    },
    {
        "id": "serious",
        "label_ru": "Серьёзно",
        "description_ru": "Коротко и по делу, без лишних слов - уважительно и конкретно.",
        "emoji": "🎯",
        "default": False,
        "doctor_safe": True,
    },
    {
        "id": "official",
        "label_ru": "Официально",
        "description_ru": "Формальные формулировки с «Вы» - для делового тона на приёме.",
        "emoji": "📋",
        "default": False,
        "doctor_safe": True,
    },
    {
        "id": "light",
        "label_ru": "С лёгкостью",
        "description_ru": "Мягкий юмор без сарказма - чтобы разрядить обстановку, но не обидеть врача.",
        "emoji": "✨",
        "default": False,
        "doctor_safe": True,
    },
]


def normalize_question_tone(value: str | None) -> QuestionTone:
    key = (value or "").strip().lower()
    return _TONE_ALIASES.get(key, DEFAULT_QUESTION_TONE)


def tone_meta(tone: str | None) -> dict[str, Any]:
    tid = normalize_question_tone(tone)
    for row in QUESTION_TONE_CATALOG:
        if row["id"] == tid:
            return dict(row)
    return dict(QUESTION_TONE_CATALOG[0])


def questions_panel_intro_ru(tone: str | None) -> str:
    tid = normalize_question_tone(tone)
    intros = {
        "friendly": "Сформулированы уважительно - чтобы врачу было легко ответить, а вам понятно.",
        "serious": "Короткие вопросы по сути - без лишнего, с уважением к времени врача.",
        "official": "Деловой тон: вежливо и чётко, как в официальном запросе на приёме.",
        "light": "С лёгкой теплотой и юмором - без насмешек, чтобы не напрягать разговор.",
    }
    return intros.get(tid, intros["friendly"])


def questions_etiquette_ru(tone: str | None) -> str:
    tid = normalize_question_tone(tone)
    hints = {
        "friendly": "Отмечайте обсуждённое - так на приёме ничего не забудете. Тон спокойный, без претензий.",
        "serious": "Задавайте по одному вопросу и слушайте ответ - так консультация пройдёт продуктивнее.",
        "official": "Обращение на «Вы» и конкретика помогают врачу дать точный ответ.",
        "light": "Лёгкая шутка уместна, если врач расположен - но суть вопроса остаётся серьёзной.",
    }
    return hints.get(tid, hints["friendly"])


def category_emoji(category_ru: str) -> str:
    return CATEGORY_EMOJI.get((category_ru or "").strip(), "💬")


def _strip_doctor_prefix(text: str) -> str:
    return re.sub(r"^доктор,?\s*", "", text.strip(), flags=re.I)


def _ensure_question(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not t.endswith("?"):
        t += "?"
    if t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _serious_style(text: str, block_id: str) -> str:
    t = _strip_doctor_prefix(text)
    t = re.sub(r"подскажите,?\s*пожалуйста,?\s*", "", t, flags=re.I)
    t = re.sub(r"не могли бы вы\s*", "Уточните: ", t, flags=re.I)
    t = re.sub(r"можно,?\s*пожалуйста,?\s*", "Уточните: ", t, flags=re.I)
    t = t.replace("  ", " ").strip()
    if not t.lower().startswith(("уточните", "нужно", "когда", "какие", "сколько", "по ")):
        if block_id == "treatment":
            t = f"По лечению: {t[0].lower() + t[1:] if len(t) > 1 else t}"
        elif block_id == "exams":
            t = f"По обследованиям: {t[0].lower() + t[1:] if len(t) > 1 else t}"
    return _ensure_question(t)


def _official_style(text: str, block_id: str) -> str:
    t = _strip_doctor_prefix(text).rstrip("?").strip()
    low = t.lower()
    if low.startswith("не могли бы"):
        return _ensure_question(t)
    if block_id == "follow_up":
        return "Не могли бы Вы сообщить дату следующего визита и перечень документов для приёма?"
    if block_id == "treatment" and "доз" in low:
        return "Не могли бы Вы уточнить режим дозирования и длительность назначенной терапии?"
    if block_id == "diagnosis":
        return "Не могли бы Вы разъяснить формулировку диагноза и его значение для дальнейшего лечения?"
    if block_id == "exams":
        return "Не могли бы Вы уточнить, какие обследования уже выполнены и какие необходимо пройти?"
    core = t[0].lower() + t[1:] if t else t
    return _ensure_question(f"Не могли бы Вы уточнить: {core}")


def _light_style(text: str, block_id: str, category_ru: str) -> str:
    t = _strip_doctor_prefix(text).rstrip("?").strip()
    low = t.lower()
    if block_id == "treatment" or category_ru == "Лечение":
        if "доз" in low or "принимать" in low:
            return _ensure_question(
                "Чтобы таблетки не путались с витаминами - подскажите, как именно мне их принимать?"
            )
        if "срок" in low or "длительност" in low:
            return _ensure_question("На сколько дней мне «прописана» терапия - чтобы не закончить раньше времени?")
    if block_id == "exams" or category_ru == "Обследования":
        if "узи" in low:
            return _ensure_question("УЗИ - это в ближайшие дни или можно не спешить?")
        return _ensure_question("По обследованиям: что уже «закрыто», а куда ещё записаться?")
    if block_id == "follow_up":
        return _ensure_question("Когда вас снова ждать на приёме - чтобы не приехать слишком рано или поздно?")
    if block_id == "diagnosis":
        return _ensure_question("Можно по-простому, «на пальцах» - что означает мой диагноз?")
    if "анализ" in low:
        return _ensure_question("В анализах есть цифры, которых нет в заключении - они учтены в плане лечения?")
    return _ensure_question(f"Извините за банальный вопрос - {t[0].lower() + t[1:] if t else 'не могли бы вы пояснить'}")


def apply_question_tone(
    text: str,
    tone: str | None,
    *,
    block_id: str = "",
    category_ru: str = "",
) -> str:
    """Перефразировать готовый вопрос под выбранный тон."""
    base = _ensure_question(text)
    if not base:
        return ""
    tid = normalize_question_tone(tone)
    bid = (block_id or "").strip().lower()
    if tid == "friendly":
        return base
    if tid == "serious":
        return _serious_style(base, bid)
    if tid == "official":
        return _official_style(base, bid)
    if tid == "light":
        return _light_style(base, bid, category_ru)
    return base


def apply_tone_to_questions(
    questions: list[dict[str, Any]],
    tone: str | None,
) -> list[dict[str, Any]]:
    """Добавить tone-поля и перефразировать text/title."""
    tid = normalize_question_tone(tone)
    meta = tone_meta(tid)
    out: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        row = dict(q)
        bid = str(row.get("block_id") or "")
        cat = str(row.get("category_ru") or "")
        raw = str(row.get("text") or "")
        styled = apply_question_tone(raw, tid, block_id=bid, category_ru=cat)
        row["text"] = styled
        row["text_base"] = raw
        row["title"] = styled.split("?")[0].strip()[:72] + ("?" if "?" in styled else "")
        row["tone"] = tid
        row["emoji"] = row.get("emoji") or category_emoji(cat)
        out.append(row)
    for row in out:
        row.setdefault("tone", tid)
    return out


def question_tones_for_api() -> list[dict[str, Any]]:
    return [dict(x) for x in QUESTION_TONE_CATALOG]
