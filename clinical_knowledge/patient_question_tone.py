"""Тон формулировок «вопросов врачу» для B2C - три чётко различимых стиля."""
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
    "calm_respectful": "serious",
    "спокойно": "serious",
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
    "Жалобы": "speech",
    "Анамнез": "history",
    "Осмотр": "stethoscope",
    "Диагноз": "dna",
    "Обследования": "scan",
    "Лечение": "pill",
    "Контроль": "calendar",
    "Анализы": "lab",
    "Протокол": "protocol",
    "Документ": "document",
    "Вопрос на приёме": "chat",
}

QUESTION_TONE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "serious",
        "label_ru": "Строго и серьёзно",
        "description_ru": "Коротко, по делу, без шуток - уважение к времени врача.",
        "emoji": "serious",
        "icon": "serious",
        "accent": "#1e3a5f",
        "default": True,
        "doctor_safe": True,
    },
    {
        "id": "official",
        "label_ru": "Официально",
        "description_ru": "Деловой стиль, обращение на «Вы» - как официальный запрос.",
        "emoji": "official",
        "icon": "official",
        "accent": "#1d4ed8",
        "default": False,
        "doctor_safe": True,
    },
    {
        "id": "playful",
        "label_ru": "Шуточно",
        "description_ru": "Креативно и с юмором про выписку и анализы - без сарказма к врачу.",
        "emoji": "playful",
        "icon": "playful",
        "accent": "#b8860b",
        "default": False,
        "doctor_safe": True,
    },
]

# intent → {tone: question} (serious / official; playful - из _PLAYFUL_VARIANTS)
_QUESTION_BANK: dict[str, dict[str, str]] = {
    "treatment_duration": {
        "serious": "На какой срок назначена терапия и когда планируется завершение курса?",
        "official": "Прошу уточнить срок назначенной терапии и дату предполагаемого окончания лечения.",
    },
    "treatment_dose": {
        "serious": "Уточните дозировку, кратность приёма и длительность назначенных препаратов.",
        "official": "Прошу разъяснить режим дозирования, время приёма и продолжительность назначенной терапии.",
    },
    "treatment_unclear": {
        "serious": "В заключении недостаточно деталей по лечению - прошу пояснить тактику.",
        "official": "Прошу дополнить раздел лечения: режим, длительность и ожидаемый эффект терапии.",
    },
    "exams_uzi": {
        "serious": "Нужно ли выполнить УЗИ и в какие сроки?",
        "official": "Прошу указать, требуется ли УЗИ и рекомендуемые сроки проведения исследования.",
    },
    "exams_oak": {
        "serious": "Требуется ли контрольный общий анализ крови с учётом последних результатов?",
        "official": "Прошу указать необходимость повторного ОАК и учёт ранее полученных показателей.",
    },
    "exams_plan": {
        "serious": "Какие обследования уже выполнены и какие необходимо пройти далее?",
        "official": "Прошу перечислить выполненные исследования и план дальнейшей диагностики.",
    },
    "exams_protocol_gap": {
        "serious": "По стандарту лечения требуется обследование, которого нет в заключении - нужно ли его пройти?",
        "official": "Прошу разъяснить необходимость обследования, предусмотренного клиническим протоколом, но не отражённого в выписке.",
    },
    "follow_up": {
        "serious": "Когда следующий визит и что подготовить к приёму?",
        "official": "Прошу указать дату следующего визита и перечень документов для приёма.",
    },
    "diagnosis_plain": {
        "serious": "Прошу объяснить диагноз простыми словами и его значение для лечения.",
        "official": "Прошу разъяснить формулировку диагноза и клиническое значение для дальнейшей тактики.",
    },
    "diagnosis_gap": {
        "serious": "В диагнозе не хватает ясности - прошу уточнить формулировку.",
        "official": "Прошу уточнить формулировку диагноза и недостающие клинические детали.",
    },
    "complaints_gap": {
        "serious": "В жалобах не отражён важный симптом - нужно ли его учесть?",
        "official": "Прошу уточнить, следует ли включить в оценку симптом, не указанный в разделе жалоб.",
    },
    "anamnesis_gap": {
        "serious": "В анамнезе не указаны важные сведения - нужно ли их дополнить?",
        "official": "Прошу уточнить, требуется ли дополнение анамнеза недостающими данными.",
    },
    "objective_gap": {
        "serious": "В объективном статусе не описан важный признак - прошу пояснить.",
        "official": "Прошу разъяснить отсутствие в осмотре клинического признака, значимого для случая.",
    },
    "localization": {
        "serious": "Уточните локализацию процесса и клиническое значение.",
        "official": "Прошу уточнить анатомическую локализацию и её значение для тактики лечения.",
    },
    "staging": {
        "serious": "На какой стадии заболевание и как это влияет на лечение?",
        "official": "Прошу указать стадию заболевания и её влияние на выбранную тактику.",
    },
    "labs_plan": {
        "serious": "Какие лабораторные исследования ещё необходимы по плану лечения?",
        "official": "Прошу указать перечень необходимых лабораторных исследований.",
    },
    "labs_missing_in_kz": {
        "serious": "В анализах есть показатели, не отражённые в заключении - учтены ли они?",
        "official": "Прошу разъяснить, учтены ли в тактике лечения показатели из бланков анализов, не указанные в выписке.",
    },
    "document_quality": {
        "serious": "Качество распознавания документа низкое - возможны ошибки в оценке. Что переснять?",
        "official": "Прошу указать, какие фрагменты заключения необходимо предоставить повторно в связи с низким качеством распознавания.",
    },
}

# Несколько уникальных шуточных формулировок на intent - без повторов в одном отчёте.
_PLAYFUL_VARIANTS: dict[str, list[str]] = {
    "treatment_duration": [
        "Рецепт на руках, а до какого числа пить - в выписке не нашёл. Подскажете дату, как в регистратуре: «до когда»?",
        "Таблетки уже дома лежат, а срок курса будто в очереди за талоном - до какого числа их принимать?",
        "Чтобы не пить «пока не забуду» - на сколько дней или недель рассчитана терапия?",
    ],
    "treatment_dose": [
        "В рецепте красивый почерк, а сколько и когда - для меня загадка. Утром, вечером, после еды - распишете простыми словами?",
        "Доза у вас в голове, у меня - баночка без инструкции. Сколько таблеток и когда, чтобы не ошибиться?",
        "Боюсь перепутать, как с очередями в разные кабинеты - напишете схему приёма для обычного человека?",
    ],
    "treatment_unclear": [
        "Раздел «Лечение» прочитал, но для меня это как объявление в поликлинике мелким шрифтом - что именно делать каждый день?",
        "Лекарства перечислены, а логика курса - нет. Это так задумано или на приёме допишете?",
        "По терапии в выписке туман - не разложите по полочкам: что, зачем и до какого момента?",
    ],
    "exams_uzi": [
        "Про УЗИ в заключении тишина - мне уже записываться или пока ждать, как талон «на потом»?",
        "УЗИ нужно срочно или можно спокойно вписать в свой график, как анализы между работой и делами?",
        "В выписке УЗИ не упомянуто - назначаем или моё «внутреннее эхо» уже достаточно для истории?",
    ],
    "exams_oak": [
        "Свежий бланк ОАК в сумке, а в КЗ про него ни слова - цифры уже учли или принести на приём отдельно?",
        "Анализ крови сдан в нашей поликлинике, стрелочки есть, в заключении - тишина. Пересдаём или эти цифры «в деле»?",
        "ОАК свежий, выписка будто без него написана - ориентируемся на последние показатели?",
    ],
    "exams_plan": [
        "Обследования в КЗ как список покупок без галочек - что уже сделано, а куда ещё записаться?",
        "Исследования названы, а что «закрыто», а что впереди - пройдёмся коротко, как по маршруту по кабинетам?",
        "Диагностика в выписке без отметок «готово» - что из этого я уже прошёл, а что только планируется?",
    ],
    "exams_protocol_gap": [
        "По протоколу Минздрава положено одно обследование, в моём КЗ - пусто. Это осознанно или мне идти сдавать?",
        "Стандарт лечения и моя выписка расходятся - как в двух разных памятках. Что для меня актуально?",
        "Клинический протокол намекает на исследование, в заключении его нет - нужно пройти или у нас другая тактика?",
    ],
    "follow_up": [
        "Когда приходить снова - через неделю, месяц или «когда самочувствие попросит»?",
        "Дата следующего визита у вас в голове, у меня - пустая строка в блокноте. Зафиксируем и что принести?",
        "Повторный приём - это «скоро», «через N дней» или «по записи, когда будет талон»?",
    ],
    "diagnosis_plain": [
        "Диагноз в заключении - как латынь в карте: звучит солидно, а что это для меня - не ясно. Объясните простыми словами?",
        "Строка «Диагноз» есть, а жить с этим знанием пока непонятно как - расшифруете по-человечески?",
        "МКБ и термины на месте, перевода «для пациента» - нет. На что мне обращать внимание в быту?",
    ],
    "diagnosis_gap": [
        "Диагноз в КЗ как смс без последней части - допишете, что именно имеется в виду?",
        "Формулировка намёком, а я люблю ясность - уточните, чтобы ночью не гуглить в панике?",
        "В выписке диагноз намечен, но не подписан до конца - доведёте до понятной формулировки?",
    ],
    "complaints_gap": [
        "Жаловался(ась) на одно, в выписке другое - мы про разное говорили или просто сократили?",
        "Мои симптомы помню, в разделе «Жалобы» - урезано. Что важно дописать, чтобы картина была полной?",
        "В КЗ жалобы короткие - не потерялось ли что-то из того, с чем я пришёл(ла) в кабинет?",
    ],
    "anamnesis_gap": [
        "Анамнез как черновик на коленке - важное выпало. Дополню устно или лучше списком на бумаге?",
        "История болезни усечена - что из прошлого критично вспомнить на приёме?",
        "В анамнезе дырка - мне восстановить хронологию или вы сами допишете по моим словам?",
    ],
    "objective_gap": [
        "Осмотр был, а в тексте половина пропала - как фото обрезали. Что из найденного важно для меня?",
        "Вы всё видели, выписка - не всё рассказала. Какие признаки при осмотре мне стоит помнить?",
        "Объективный статус в КЗ урезан - чего не хватает в описании для полной картины?",
    ],
    "localization": [
        "Где именно «сидит» проблема - чтобы не гуглить всё подряд, а знать точку?",
        "Локализация размыта - уточните орган или зону простыми словами?",
        "Процесс где-то внутри, но адрес не указан - покажете на схеме или словами?",
    ],
    "staging": [
        "На какой стадии сейчас - начало пути, середина или уже «серьёзный участок», где меняется лечение?",
        "Стадия в КЗ не названа - это ранняя глава или уже поворот, от которого зависит тактика?",
        "Чтобы не гадать: болезнь на старте или уже там, где нужен другой подход?",
    ],
    "labs_plan": [
        "Какие анализы ещё впереди - хочу сдать за один заход, как все дела в поликлинике за один день?",
        "Лабораторный план без дат - что срочно, а что можно вместе с контрольным визитом?",
        "Очередь из пробирок - подскажете порядок, чтобы не ездить в лабораторию каждый второй день?",
    ],
    "labs_missing_in_kz": [
        "Бланк из лаборатории с цифрами и стрелочками, а заключение молчит - учли при лечении или принести и зачитать?",
        "Анализы на руках, в КЗ про них ни слова - это нормально или стоит напомнить на приёме?",
        "Лаборатория выдала результат, выписка будто без него - показатели уже в тактике или ждут очереди?",
        "В бланке анализов больше цифр, чем в тексте заключения - они уже повлияли на назначения?",
    ],
    "document_quality": [
        "Фото смазалось, как снимок из автобуса - что переснять, чтобы вы прочитали КЗ без догадок?",
        "Качество снимка слабое, буквы пляшут - какие страницы сфотографировать заново?",
        "Загрузилось криво - подскажете, что переснять или лучше принести PDF из клиники?",
    ],
    "exams_timing": [
        "Когда удобнее пройти {anchor} и куда записаться, чтобы не бегать лишний раз по кабинетам?",
        "В выписке есть {anchor}, а даты нет - это срочно или можно вписать в свой график?",
        "{anchor} назначены - когда записываться и нужен ли направляющий талон?",
    ],
    "diagnosis_uncertain": [
        "Диагноз в выписке с вопросом - какие обследования его подтвердят или опровергнут?",
        "Формулировка диагноза пока предварительная - что сдавать или проходить для точности?",
        "Строка диагноза выглядит не окончательной - какой план, чтобы понять картину?",
    ],
    "treatment_order": [
        "Назначено несколько препаратов ({anchor}) - в каком порядке принимать и можно ли вместе?",
        "{anchor} в одном рецепте - утром всё сразу или по очереди, без путаницы дома?",
        "Боюсь перепутать таблетки из {anchor} - распишете схему приёма простыми словами?",
    ],
    "clarify": [
        "По {anchor} в выписке не всё ясно - поясните на приёме простыми словами?",
        "В заключении {anchor} - можно разложить по шагам, что это значит для меня?",
        "Про {anchor} остался вопрос - уточните, пожалуйста, без медицинских терминов?",
    ],
}

_PLAYFUL_META_MARKERS = (
    "намёк",
    "намек",
    "не для претензии - хочу ясности",
    "медицинских загадок",
)


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
        "playful": "Тёплые вопросы про выписку и анализы - по-белорусски: вежливо, с лёгкой улыбкой, без претензий к врачу.",
    }[tid]


def questions_etiquette_ru(tone: str | None) -> str:
    tid = normalize_question_tone(tone)
    return {
        "serious": "Задавайте по одному вопросу. Отмечайте обсуждённое - ничего не забудете на приёме.",
        "official": "Сохраняйте деловой тон. Конкретика помогает врачу дать точный ответ.",
        "playful": "Шутка - чтобы разрядить очередь в коридоре, не чтобы спорить. Если врач занят, начните с одного серьёзного вопроса.",
    }[tid]


def category_emoji(category_ru: str) -> str:
    """Идентификатор иконки категории (для UI)."""
    return CATEGORY_EMOJI.get((category_ru or "").strip(), "chat")


def _format_playful_variant(template: str, anchor: str) -> str:
    a = (anchor or "").strip().rstrip(".")
    if "{anchor}" in template:
        return template.format(anchor=a or "назначение")
    if a and a.lower() not in template.lower():
        return f"{template.rstrip('?')} ({a})?"
    return template


def is_playful_meta_template(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _PLAYFUL_META_MARKERS)


def _pick_playful_text(
    intent: str | None,
    *,
    slot: int = 0,
    used: set[str] | None = None,
    plain_context: str = "",
) -> str:
    """Выбрать уникальную шуточную формулировку по intent и слоту."""
    used = used or set()
    key = intent or ""
    variants = _PLAYFUL_VARIANTS.get(key) or []
    if variants:
        for i in range(len(variants)):
            candidate = _format_playful_variant(variants[(slot + i) % len(variants)], plain_context)
            norm = candidate.lower().strip()
            if norm not in used:
                return candidate
        return _format_playful_variant(variants[slot % len(variants)], plain_context)
    bank = _QUESTION_BANK.get(key) or {}
    return bank.get("playful") or ""


def _playful_soft_fallback(gap: str, block_name: str, *, fallback_text: str = "") -> str:
    preset = (fallback_text or "").strip()
    if preset:
        return preset
    g = (gap or "").strip().rstrip(".")
    name = block_name or "разделу"
    if g:
        return _ensure_question(f"На приёме можно уточнить по «{name}»: {g}")
    return _ensure_question(f"Про «{name}» в выписке остался вопрос - поясните простыми словами?")


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
    *,
    playful_slot: int = 0,
    playful_used: set[str] | None = None,
    fallback_text: str = "",
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
    return _playful_soft_fallback(g, name, fallback_text=fallback_text)


def render_doctor_question(
    *,
    gap: str = "",
    comment: str = "",
    block_id: str = "",
    block_name: str = "",
    category_ru: str = "",
    tone: str | None,
    intent: str | None = None,
    playful_slot: int = 0,
    playful_used: set[str] | None = None,
    fallback_text: str = "",
    plain_context: str = "",
) -> tuple[str, str | None]:
    """Сформировать вопрос в выбранном тоне. Возвращает (text, intent)."""
    tid = normalize_question_tone(tone)
    preset = (fallback_text or "").strip()
    raw = (comment or gap or "").strip()
    kind = "comment" if (comment or "").strip() else "gap"
    key = intent or detect_question_intent(raw, block_id, kind=kind)

    if key == "document_quality" or (block_id == "limitations" and "качество" in raw.lower()):
        key = "document_quality"

    has_template = key and (key in _QUESTION_BANK or key in _PLAYFUL_VARIANTS)
    if has_template:
        if tid == "playful":
            text = _pick_playful_text(
                key,
                slot=playful_slot,
                used=playful_used,
                plain_context=plain_context,
            )
            if not text and preset:
                text = preset
        else:
            text = (_QUESTION_BANK.get(key) or {}).get(tid) or (_QUESTION_BANK.get(key) or {}).get("serious", "")
        if text:
            if tid == "playful" and is_playful_meta_template(text) and preset:
                text = preset
            return _ensure_question(text), key

    if preset:
        return _ensure_question(preset), key

    if not raw:
        return "", key

    if raw.endswith("?"):
        base = _ensure_question(raw)
        if tid == "official" and "вы" not in base.lower():
            return _ensure_question(f"Прошу уточнить: {base.rstrip('?').lower()}"), key
        if tid == "playful" and "извините" not in base.lower():
            return _ensure_question(f"Можно честно спросить: {base.rstrip('?').lower()}?"), key
        return base, key

    text = _generic_by_tone(
        raw, block_name, block_id, tid,
        playful_slot=playful_slot,
        playful_used=playful_used,
        fallback_text=preset,
    )
    return text, key


def apply_tone_to_questions(
    questions: list[dict[str, Any]],
    tone: str | None,
) -> list[dict[str, Any]]:
    tid = normalize_question_tone(tone)
    out: list[dict[str, Any]] = []
    intent_slots: dict[str, int] = {}
    playful_used: set[str] = set()
    for q in questions:
        if not isinstance(q, dict):
            continue
        row = dict(q)
        bid = str(row.get("block_id") or "")
        cat = str(row.get("category_ru") or "")
        name = cat or bid
        pre_intent = row.get("intent") or detect_question_intent(
            str(row.get("source_comment") or row.get("source_gap") or ""),
            bid,
            kind="comment" if row.get("source_comment") else "gap",
        )
        slot = intent_slots.get(pre_intent or "", 0)
        if pre_intent:
            intent_slots[pre_intent] = slot + 1
        preset = str(row.get("text") or "").strip()
        plain_ctx = str(row.get("plain_context") or "").strip()
        styled, intent = render_doctor_question(
            gap=str(row.get("source_gap") or ""),
            comment=str(row.get("source_comment") or ""),
            block_id=bid,
            block_name=name,
            category_ru=cat,
            tone=tid,
            intent=row.get("intent"),
            playful_slot=slot,
            playful_used=playful_used,
            fallback_text=preset,
            plain_context=plain_ctx,
        )
        if not styled and preset:
            styled, intent = render_doctor_question(
                gap=preset,
                block_id=bid,
                block_name=name,
                category_ru=cat,
                tone=tid,
                intent=row.get("intent"),
                playful_slot=slot,
                playful_used=playful_used,
                fallback_text=preset,
                plain_context=plain_ctx,
            )
        if styled and tid == "playful":
            playful_used.add(styled.lower().strip())
        row["text"] = styled
        row["intent"] = intent
        row["title"] = styled.split("?")[0].strip()[:72] + ("?" if "?" in styled else "")
        row["tone"] = tid
        icon_id = category_emoji(cat)
        row["icon"] = icon_id
        row["emoji"] = icon_id
        out.append(row)
    return out


def question_tones_for_api() -> list[dict[str, Any]]:
    return [dict(x) for x in QUESTION_TONE_CATALOG]
