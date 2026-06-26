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
        "В выписке рецепт есть, а финал курса как у сериала без последней серии - на сколько недель мне растянуть лечение?",
        "Срок терапии в заключении спрятался лучше скидки в мелком шрифте - подскажете дату, когда можно отпраздновать «выписку с таблеток»?",
        "Таблетки уже на полке, а календарь молчит: лечение до конкретной даты или «пока организм не скажет спасибо»?",
    ],
    "treatment_dose": [
        "Доза в голове врача, а у меня дома просто баночка - сколько штук, до или после еды и можно ли запивать чаем?",
        "Режим приёма в КЗ описан намёком: утром, вечером или по будильнику, который я сам придумываю?",
        "Боюсь сыграть в «угадай дозу» - пропишете схему так, чтобы даже я не ошибся между завтраком и ужином?",
    ],
    "treatment_unclear": [
        "Раздел «Лечение» прочитан, но смысл ускользнул - как рецепт, написанный стихами. Расшифруете тактику для обычного человека?",
        "В заключении лекарства перечислены, а логика курса - нет. Это минимализм или мне что-то донести устно на приёме?",
        "По терапии в выписке загадка посложнее кроссворда - не соберёте картинку: что, зачем и до какого момента?",
    ],
    "exams_uzi": [
        "УЗИ в заключении не светится - мне уже бронировать кабинет или пока жить в сюжете без финальной сцены?",
        "Аппарат УЗИ ждёт меня на этой неделе или это исследование из категории «когда-нибудь потом»?",
        "В выписке про УЗИ тишина, а в голове вопросы шумят - назначаем или моё «внутреннее эхо» уже достаточно?",
    ],
    "exams_oak": [
        "Свежий бланк ОАК лежит в сумке, а заключение его не цитирует - цифры уже в сюжете лечения или отдельная глава?",
        "Анализ крови сдан, стрелочки нарисованы, в КЗ - ни слова. Пересдаём или эти результаты уже «в деле»?",
        "ОАК как свежий выпуск новостей, а выписка будто вышла до редакции - учитываем последние показатели?",
    ],
    "exams_plan": [
        "Обследования в КЗ как список покупок без галочек - что уже закрыто, а куда ещё записываться в квест на здоровье?",
        "В заключении исследования названы, а статус «сделано / впереди» спрятан - пройдёмся по чек-листу вместе?",
        "Диагностика в выписке как меню без отметок «заказано» - что из этого я уже прошёл, а что только в планах?",
    ],
    "exams_protocol_gap": [
        "Протокол Минздрава намекает на одно обследование, в моём КЗ - тишина. Догоняем стандарт или у нас свой сценарий?",
        "По клиническому протоколу положено исследование, в выписке его нет - это осознанный пропуск или мне идти сдавать?",
        "Стандарт лечения и моё заключение расходятся как две версии одного фильма - какую смотреть пациенту?",
    ],
    "follow_up": [
        "Когда снова приходить - через неделю, месяц или когда организм сам постучится в календарь?",
        "Следующий визит в голове врача, а у меня пустая строка в блокноте - зафиксируем дату и что принести?",
        "Повторный приём - это «скоро», «через N дней» или «по самочувствию, но не затягивайте»?",
    ],
    "diagnosis_plain": [
        "Диагноз в заключении звучит как название фильма на латинском - можно субтитры на русском и главную мысль для меня?",
        "Строка «Диагноз» прочитана, а жить с этим знанием пока непонятно как - объясните простыми словами, что это значит?",
        "МКБ и медицинские термины есть, перевода на человеческий - нет. Расшифруете, на что мне обращать внимание?",
    ],
    "diagnosis_gap": [
        "Диагноз в КЗ как SMS с обрезанным текстом - не хватает концовки. Допишете, что именно имеется в виду?",
        "Формулировка диагноза намёком, а я люблю ясность - уточните, чтобы не гуглить ночью в панике?",
        "В выписке диагноз намечен карандашом, а не подписан чернилами - доведёте формулировку до понятной?",
    ],
    "complaints_gap": [
        "Жаловался(ась) на одно, в выписке записано иначе - редактор сократил или мы говорили о разном?",
        "Мои симптомы в голове ясные, в разделе «Жалобы» - урезанные. Что важно дописать, чтобы картина была полной?",
        "В КЗ жалобы как тизер без спойлеров - не потерялось ли что-то важное из того, с чем я пришёл(ла)?",
    ],
    "anamnesis_gap": [
        "Анамнез в заключении как черновик - важные детали выпали. Дополню устно или принести список на бумаге?",
        "История болезни в КЗ усечена, будто лимит символов - что из прошлого критично вспомнить на приёме?",
        "В анамнезе дырка размером с важное событие - мне восстановить хронологию или вы сами допишете?",
    ],
    "objective_gap": [
        "Осмотр был, а в тексте заключения половина пропала - как фото обрезали. Что из найденного важно для меня?",
        "Врач всё видел, выписка - не всё рассказала. Какие признаки при осмотре я должен(на) помнить?",
        "Объективный статус в КЗ как трейлер без ключевых кадров - чего не хватает в описании?",
    ],
    "localization": [
        "Где именно «поселилась» проблема - чтобы не гуглить всё тело подряд, а знать точку на карте?",
        "Локализация в заключении размыта - уточните орган/зону, чтобы я не фантазировал лишнего?",
        "Процесс где-то внутри, но адрес не указан - покажете на схеме или словами, где искать?",
    ],
    "staging": [
        "На какой стадии болезнь сейчас - лёгкий уровень, средний или финальный босс, с которым мы справимся?",
        "Стадия в КЗ не названа явно - это ранняя глава или уже середина истории, от которой зависит лечение?",
        "Чтобы не строить догадки: болезнь на старте пути или уже на повороте, где меняется тактика?",
    ],
    "labs_plan": [
        "Какие анализы ещё впереди - хочу собрать их в один поход в лабораторию, как продукты в одну корзину?",
        "Лабораторный план в заключении как список дел без дат - что сдавать срочно, а что можно вместе с контролем?",
        "Передо мной очередь из пробирок - подскажете порядок, чтобы не кататься в лабораторию каждый второй день?",
    ],
    "labs_missing_in_kz": [
        "Принёс бланки с цифрами и стрелочками, а заключение их не цитирует - анализы уже в плане лечения или живут в параллельной вселенной?",
        "Лаборатория выдала красные и зелёные маркеры, в выписке - ни слова. Это осознанное молчание или стоит напомнить о результатах?",
        "Анализы на руках, в КЗ про них тишина: показатели учтены при назначении терапии или мне их зачитать врачу вслух?",
        "Бланк из лаборатории богаче текста заключения по цифрам - они уже повлияли на лечение или ждут своего дебюта в выписке?",
    ],
    "document_quality": [
        "Фото заключения как портрет в движении - что переснять, чтобы вы прочитали КЗ без догадок и фантазии?",
        "Снимок КЗ получился смазанным, будто сделан на бегу - какие страницы сфотографировать заново для точной оценки?",
        "Качество загрузки низкое, буквы пляшут - подскажете, какие фрагменты выписки прислать чётко?",
    ],
}

_PLAYFUL_GENERIC: list[str] = [
    "По разделу «{name}» в выписке намёк вместо ответа: {gap} - не разложите по полочкам на приёме?",
    "В КЗ про «{name}» написано загадкой: {gap}. Можно версию для пациента без медицинского детектива?",
    "Заключение молчит там, где я жду ясности ({name}): {gap} - проясните, пожалуйста?",
    "Раздел «{name}» прочитан, но {gap} осталось без расшифровки - допишете смысл для меня?",
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
    return {
        "serious": "Короткие вопросы по сути - без лишних слов, с уважением к врачу.",
        "official": "Формальный деловой тон: чётко, на «Вы», как официальный запрос на приёме.",
        "playful": "Креативные вопросы про выписку, анализы и то, что в КЗ написано намёком - с юмором, но по делу.",
    }[tid]


def questions_etiquette_ru(tone: str | None) -> str:
    tid = normalize_question_tone(tone)
    return {
        "serious": "Задавайте по одному вопросу. Отмечайте обсуждённое - ничего не забудете на приёме.",
        "official": "Сохраняйте деловой тон. Конкретика помогает врачу дать точный ответ.",
        "playful": "Шутка - для тепла, не для спора. Если врач занят, начните с серьёзного вопроса.",
    }[tid]


def category_emoji(category_ru: str) -> str:
    """Идентификатор иконки категории (для UI)."""
    return CATEGORY_EMOJI.get((category_ru or "").strip(), "chat")


def _pick_playful_text(
    intent: str | None,
    *,
    slot: int = 0,
    used: set[str] | None = None,
) -> str:
    """Выбрать уникальную шуточную формулировку по intent и слоту."""
    used = used or set()
    key = intent or ""
    variants = _PLAYFUL_VARIANTS.get(key) or []
    if variants:
        for i in range(len(variants)):
            candidate = variants[(slot + i) % len(variants)]
            norm = candidate.lower().strip()
            if norm not in used:
                return candidate
        return variants[slot % len(variants)]
    bank = _QUESTION_BANK.get(key) or {}
    return bank.get("playful") or ""


def _pick_playful_generic(gap: str, block_name: str, slot: int, used: set[str]) -> str:
    g = (gap or "").strip().rstrip(".")
    name = block_name or "разделу"
    for i in range(len(_PLAYFUL_GENERIC)):
        tpl = _PLAYFUL_GENERIC[(slot + i) % len(_PLAYFUL_GENERIC)]
        candidate = tpl.format(name=name, gap=g)
        if candidate.lower().strip() not in used:
            return candidate
    return _PLAYFUL_GENERIC[slot % len(_PLAYFUL_GENERIC)].format(name=name, gap=g)


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
    used = playful_used or set()
    return _ensure_question(_pick_playful_generic(g, name, playful_slot, used))


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
) -> tuple[str, str | None]:
    """Сформировать вопрос в выбранном тоне. Возвращает (text, intent)."""
    tid = normalize_question_tone(tone)
    raw = (comment or gap or "").strip()
    kind = "comment" if (comment or "").strip() else "gap"
    key = intent or detect_question_intent(raw, block_id, kind=kind)

    if key == "document_quality" or (block_id == "limitations" and "качество" in raw.lower()):
        key = "document_quality"

    if key and key in _QUESTION_BANK:
        if tid == "playful":
            text = _pick_playful_text(key, slot=playful_slot, used=playful_used)
        else:
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

    text = _generic_by_tone(
        raw, block_name, block_id, tid,
        playful_slot=playful_slot,
        playful_used=playful_used,
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
        )
        if not styled and row.get("text"):
            styled, intent = render_doctor_question(
                gap=str(row.get("text") or ""),
                block_id=bid,
                block_name=name,
                category_ru=cat,
                tone=tid,
                playful_slot=slot,
                playful_used=playful_used,
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
