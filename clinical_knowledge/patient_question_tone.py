"""Тон формулировок «вопросов врачу» для B2C — три чётко различимых стиля."""
from __future__ import annotations

import re
from typing import Any, Literal

QuestionTone = Literal["playful", "official", "serious"]

DEFAULT_QUESTION_TONE: QuestionTone = "serious"

_TONE_ALIASES: dict[str, QuestionTone] = {
    "playful": "playful",
    "шуточно": "playful",
    "юмор": "playful",
    "humor": "playful",
    "light": "playful",
    "лёгкий": "playful",
    "легкий": "playful",
    "креатив": "playful",
    "official": "official",
    "официально": "official",
    "formal": "official",
    "деловой": "official",
    "serious": "serious",
    "серьёзно": "serious",
    "серьезно": "serious",
    "строго": "serious",
    # legacy → новые тоны
    "friendly": "serious",
    "дружелюбно": "serious",
    "warm": "serious",
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
        "id": "serious",
        "label_ru": "Строго и серьёзно",
        "description_ru": "Коротко, по делу, без шуток - уважение к времени врача.",
        "emoji": "🎯",
        "accent": "#1e3a5f",
        "default": True,
        "doctor_safe": True,
    },
    {
        "id": "official",
        "label_ru": "Официально",
        "description_ru": "Деловой стиль, обращение на «Вы» - как официальный запрос.",
        "emoji": "📋",
        "accent": "#1d4ed8",
        "default": False,
        "doctor_safe": True,
    },
    {
        "id": "playful",
        "label_ru": "Шуточно",
        "description_ru": "С лёгким юмором и креативом - без сарказма и претензий к врачу.",
        "emoji": "✨",
        "accent": "#d97706",
        "default": False,
        "doctor_safe": True,
    },
]

# intent → {tone: question}
_QUESTION_BANK: dict[str, dict[str, str]] = {
    "treatment_duration": {
        "serious": "На какой срок назначена терапия и когда планируется завершение курса?",
        "official": "Прошу уточнить срок назначенной терапии и дату предполагаемого окончания лечения.",
        "playful": "Чтобы не принимать таблетки «до пенсии» - до какого числа мне их пить?",
    },
    "treatment_dose": {
        "serious": "Уточните дозировку, кратность приёма и длительность назначенных препаратов.",
        "official": "Прошу разъяснить режим дозирования, время приёма и продолжительность назначенной терапии.",
        "playful": "Боюсь перепутать дозы - сколько и когда именно мне принимать препараты?",
    },
    "treatment_unclear": {
        "serious": "В заключении недостаточно деталей по лечению - прошу пояснить тактику.",
        "official": "Прошу дополнить раздел лечения: режим, длительность и ожидаемый эффект терапии.",
        "playful": "По лечению в выписке загадка посложнее кроссворда - не разберёте для меня?",
    },
    "exams_uzi": {
        "serious": "Нужно ли выполнить УЗИ и в какие сроки?",
        "official": "Прошу указать, требуется ли УЗИ и рекомендуемые сроки проведения исследования.",
        "playful": "УЗИ - это «срочно бежать» или можно спокойно записаться на следующей неделе?",
    },
    "exams_oak": {
        "serious": "Требуется ли контрольный общий анализ крови с учётом последних результатов?",
        "official": "Прошу указать необходимость повторного ОАК и учёт ранее полученных показателей.",
        "playful": "ОАК уже «устарел» или мои последние цифры в расчёт взяты?",
    },
    "exams_plan": {
        "serious": "Какие обследования уже выполнены и какие необходимо пройти далее?",
        "official": "Прошу перечислить выполненные исследования и план дальнейшей диагностики.",
        "playful": "Что из обследований уже «закрыто галочкой», а куда мне ещё записываться?",
    },
    "exams_protocol_gap": {
        "serious": "По стандарту лечения требуется обследование, которого нет в заключении - нужно ли его пройти?",
        "official": "Прошу разъяснить необходимость обследования, предусмотренного клиническим протоколом, но не отражённого в выписке.",
        "playful": "Протокол намекает на обследование, а в выписке тишина - мне его проходить?",
    },
    "follow_up": {
        "serious": "Когда следующий визит и что подготовить к приёму?",
        "official": "Прошу указать дату следующего визита и перечень документов для приёма.",
        "playful": "Когда снова приезжать - чтобы не застать вас врасплох и не приехать зря?",
    },
    "diagnosis_plain": {
        "serious": "Прошу объяснить диагноз простыми словами и его значение для лечения.",
        "official": "Прошу разъяснить формулировку диагноза и клиническое значение для дальнейшей тактики.",
        "playful": "Можно «перевести» диагноз с медицинского на человеческий - что он значит для меня?",
    },
    "diagnosis_gap": {
        "serious": "В диагнозе не хватает ясности - прошу уточнить формулировку.",
        "official": "Прошу уточнить формулировку диагноза и недостающие клинические детали.",
        "playful": "Диагноз звучит как заклинание - расшифруете, пожалуйста?",
    },
    "complaints_gap": {
        "serious": "В жалобах не отражён важный симптом - нужно ли его учесть?",
        "official": "Прошу уточнить, следует ли включить в оценку симптом, не указанный в разделе жалоб.",
        "playful": "Я жаловался(ась) на одно, а в выписке другого нет - это нормально?",
    },
    "anamnesis_gap": {
        "serious": "В анамнезе не указаны важные сведения - нужно ли их дополнить?",
        "official": "Прошу уточнить, требуется ли дополнение анамнеза недостающими данными.",
        "playful": "В анамнезе дырка - мне что-то вспомнить и дозаписать?",
    },
    "objective_gap": {
        "serious": "В объективном статусе не описан важный признак - прошу пояснить.",
        "official": "Прошу разъяснить отсутствие в осмотре клинического признака, значимого для случая.",
        "playful": "В осмотре чего-то не хватает - это упустили или мне не показали?",
    },
    "localization": {
        "serious": "Уточните локализацию процесса и клиническое значение.",
        "official": "Прошу уточнить анатомическую локализацию и её значение для тактики лечения.",
        "playful": "Где именно у меня «сидит» проблема - покажете на карте тела?",
    },
    "staging": {
        "serious": "На какой стадии заболевание и как это влияет на лечение?",
        "official": "Прошу указать стадию заболевания и её влияние на выбранную тактику.",
        "playful": "На каком «уровне сложности» сейчас болезнь - чтобы понимать масштаб?",
    },
    "labs_plan": {
        "serious": "Какие лабораторные исследования ещё необходимы по плану лечения?",
        "official": "Прошу указать перечень необходимых лабораторных исследований.",
        "playful": "Какие анализы ещё «в очереди» - чтобы сдать их за один заход?",
    },
    "labs_missing_in_kz": {
        "serious": "В анализах есть показатели, не отражённые в заключении - учтены ли они?",
        "official": "Прошу разъяснить, учтены ли в тактике лечения показатели из бланков анализов, не указанные в выписке.",
        "playful": "В анализах цифры есть, в заключении - тишина. Они уже «в деле» или про них забыли?",
    },
    "document_quality": {
        "serious": "Качество распознавания документа низкое - возможны ошибки в оценке. Что переснять?",
        "official": "Прошу указать, какие фрагменты заключения необходимо предоставить повторно в связи с низким качеством распознавания.",
        "playful": "Фото получилось смазанным - что переснять, чтобы вы ничего не упустили?",
    },
}


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
    return {
        "serious": "Короткие вопросы по сути - без лишних слов, с уважением к врачу.",
        "official": "Формальный деловой тон: чётко, на «Вы», как официальный запрос на приёме.",
        "playful": "С лёгким юмором и образами - чтобы разрядить разговор, но суть вопроса серьёзная.",
    }[tid]


def questions_etiquette_ru(tone: str | None) -> str:
    tid = normalize_question_tone(tone)
    return {
        "serious": "Задавайте по одному вопросу. Отмечайте обсуждённое - ничего не забудете на приёме.",
        "official": "Сохраняйте деловой тон. Конкретика помогает врачу дать точный ответ.",
        "playful": "Шутка - для тепла, не для спора. Если врач занят, начните с серьёзного вопроса.",
    }[tid]


def category_emoji(category_ru: str) -> str:
    return CATEGORY_EMOJI.get((category_ru or "").strip(), "💬")


def _ensure_question(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not t.endswith("?"):
        t += "?"
    if t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def detect_question_intent(
    text: str,
    block_id: str = "",
    *,
    kind: str = "gap",
) -> str | None:
    """Определить шаблон вопроса по тексту пробела и разделу КЗ."""
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    bid = (block_id or "").strip().lower()

    if kind == "comment" or len(raw) > 80:
        if "доз" in low and ("не детализ" in low or "не указан" in low):
            return "treatment_dose"
        if "мало детал" in low or "кратко" in low:
            return "treatment_unclear" if bid == "treatment" else "exams_plan"
        if "не указан" in low and bid == "diagnosis":
            return "diagnosis_gap"
        if bid == "exams" and ("мало" in low or "не распознан" in low):
            return "exams_plan"

    if re.search(r"длительност", low) and re.search(r"терап|лечен|при[её]м", low):
        return "treatment_duration"
    if re.search(r"доз", low):
        return "treatment_dose"
    if re.search(r"узи|ультразвук", low):
        return "exams_uzi"
    if re.search(r"\bоак\b|анализ крови", low):
        return "exams_oak"
    if re.search(r"контрол|наблюден|повторн", low):
        return "follow_up"
    if re.search(r"локализац", low):
        return "localization"
    if re.search(r"стади", low):
        return "staging"
    if re.search(r"обязательн", low) and re.search(r"лаборатор|исследован", low):
        return "labs_plan"
    if re.search(r"протокол", low) and re.search(r"обследован|узи|анализ", low):
        return "exams_protocol_gap"
    if "стандарт" in low and "обследован" in low:
        return "exams_protocol_gap"
    if "анализ" in low and ("нет в заключении" in low or "не назван" in low):
        return "labs_missing_in_kz"

    if bid == "treatment":
        return "treatment_unclear"
    if bid == "exams":
        return "exams_plan"
    if bid == "diagnosis":
        return "diagnosis_gap"
    if bid == "follow_up":
        return "follow_up"
    if bid == "complaints":
        return "complaints_gap"
    if bid == "anamnesis":
        return "anamnesis_gap"
    if bid == "objective_status":
        return "objective_gap"
    if bid == "limitations" and "качество" in low:
        return "document_quality"
    return None


def _generic_by_tone(
    gap: str,
    block_name: str,
    block_id: str,
    tone: QuestionTone,
) -> str:
    g = (gap or "").strip().rstrip(".")
    name = block_name or "разделу"
    if tone == "serious":
        if block_id == "treatment":
            return _ensure_question(f"По лечению неясно: {g}. Прошу пояснить.")
        if block_id == "exams":
            return _ensure_question(f"По обследованиям: {g}. Это уже выполнено?")
        return _ensure_question(f"По разделу «{name}»: прошу уточнить - {g}.")
    if tone == "official":
        return _ensure_question(f"Прошу уточнить по разделу «{name}»: {g}.")
    return _ensure_question(f"Извините за банальный вопрос - по «{name}»: {g}, не могли бы пояснить?")


def render_doctor_question(
    *,
    gap: str = "",
    comment: str = "",
    block_id: str = "",
    block_name: str = "",
    category_ru: str = "",
    tone: str | None,
    intent: str | None = None,
) -> tuple[str, str | None]:
    """Сформировать вопрос в выбранном тоне. Возвращает (text, intent)."""
    tid = normalize_question_tone(tone)
    raw = (comment or gap or "").strip()
    kind = "comment" if (comment or "").strip() else "gap"
    key = intent or detect_question_intent(raw, block_id, kind=kind)

    if key == "document_quality" or (block_id == "limitations" and "качество" in raw.lower()):
        key = "document_quality"

    if key and key in _QUESTION_BANK:
        text = _QUESTION_BANK[key].get(tid) or _QUESTION_BANK[key]["serious"]
        return _ensure_question(text), key

    if not raw:
        return "", key

    if raw.endswith("?"):
        base = _ensure_question(raw)
        if tid == "official" and "вы" not in base.lower():
            return _ensure_question(f"Прошу уточнить: {base.rstrip('?').lower()}"), key
        if tid == "playful" and "извините" not in base.lower():
            return _ensure_question(f"Можно честно спросить: {base.rstrip('?').lower()}?"), key
        return base, key

    text = _generic_by_tone(raw, block_name, block_id, tid)
    return text, key


def apply_tone_to_questions(
    questions: list[dict[str, Any]],
    tone: str | None,
) -> list[dict[str, Any]]:
    tid = normalize_question_tone(tone)
    out: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        row = dict(q)
        bid = str(row.get("block_id") or "")
        cat = str(row.get("category_ru") or "")
        name = cat or bid
        styled, intent = render_doctor_question(
            gap=str(row.get("source_gap") or ""),
            comment=str(row.get("source_comment") or ""),
            block_id=bid,
            block_name=name,
            category_ru=cat,
            tone=tid,
            intent=row.get("intent"),
        )
        if not styled and row.get("text"):
            styled, intent = render_doctor_question(
                gap=str(row.get("text") or ""),
                block_id=bid,
                block_name=name,
                category_ru=cat,
                tone=tid,
            )
        row["text"] = styled
        row["intent"] = intent
        row["title"] = styled.split("?")[0].strip()[:72] + ("?" if "?" in styled else "")
        row["tone"] = tid
        row["emoji"] = row.get("emoji") or category_emoji(cat)
        out.append(row)
    return out


def question_tones_for_api() -> list[dict[str, Any]]:
    return [dict(x) for x in QUESTION_TONE_CATALOG]
