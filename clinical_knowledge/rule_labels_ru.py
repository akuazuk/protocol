"""Русские подписи для rule_id, decision и сообщений rule_checker (UI/отчёты)."""
from __future__ import annotations

import re

from .condition_registry import CONDITION_BY_ID

# Человекочитаемые названия нозологий (приоритет над маркерами из реестра).
_CONDITION_TITLE_RU: dict[str, str] = {
    "gerd": "ГЭРБ",
    "gastritis": "Гастрит",
    "functional_dyspepsia": "Функциональная диспепсия",
    "peptic_ulcer": "Язвенная болезнь",
    "crohn": "Болезнь Крона",
    "ulcerative_colitis": "Язвенный колит",
    "celiac": "Целиакия",
    "neoplasm": "Новообразование",
    "carcinoma": "Злокачественное новообразование",
    "sle": "Системная красная волчанка",
}

_RULE_KIND_RU: dict[str, str] = {
    "diagnosis_formula": "Структура формулировки диагноза",
    "diagnosis_structure_rule": "Структура формулировки диагноза",
    "numbered_diagnosis_formula": "Структура формулировки диагноза",
    "generic_diagnosis_formula": "Структура формулировки диагноза",
    "diagnostic_criterion": "Диагностический критерий",
    "diagnostic_criterion_rule": "Диагностический критерий",
    "diagnostic_criteria": "Диагностические критерии",
    "histo_criterion": "Гистологическое описание",
    "gfd_mention": "Безглютеновая диета",
    "population_guard": "Применимость по возрастной группе",
    "population_mismatch": "Несоответствие возрастной группе",
    "age_applicability_rule": "Применимость по возрасту",
    "required_exam": "Обязательное обследование",
    "required_exam_rule": "Обязательное обследование",
    "conditional_exam_rule": "Обследование по показаниям",
    "keyword_presence": "Ключевое требование протокола",
    "drug_rule": "Лекарственная терапия",
    "informational_rule": "Справочное правило",
}

_DECISION_RU: dict[str, str] = {
    "satisfied": "Выполнено",
    "satisfied_by_recommendation": "Назначено в рекомендациях",
    "missing": "Не выполнено",
    "not_applicable": "Не применимо",
    "manual_review": "Нужна ручная проверка",
    "unknown": "Не определено",
}

_FOUND_STATUS_RU: dict[str, str] = {
    "performed": "Выполнено",
    "recommended": "Рекомендовано",
    "mentioned": "Упомянуто в КЗ",
    "not_found": "Не найдено",
    "not_applicable": "Не применимо",
    "unknown": "Не определено",
}

_RULE_TYPE_RU: dict[str, str] = {
    "diagnosis_structure_rule": "Структура диагноза",
    "diagnostic_criterion_rule": "Диагностический критерий",
    "required_exam_rule": "Обязательное обследование",
    "age_applicability_rule": "Возрастная применимость",
    "informational_rule": "Справочно",
}

_POPULATION_RU: dict[str, str] = {
    "adult": "взрослые",
    "child": "дети",
    "newborn": "новорождённые",
    "any": "любая аудитория",
    "unknown": "не определена",
}


def condition_title_ru(condition_id: str) -> str:
    if not condition_id:
        return "Нозология"
    if condition_id in _CONDITION_TITLE_RU:
        return _CONDITION_TITLE_RU[condition_id]
    cdef = CONDITION_BY_ID.get(condition_id)
    if cdef and cdef.text_markers:
        marker = cdef.text_markers[0]
        return marker[:1].upper() + marker[1:]
    return condition_id.replace("_", " ")


def extract_condition_id(rule_id: str) -> str | None:
    """Извлекает condition_id из технического rule_id."""
    if not rule_id:
        return None
    rid = rule_id.lower().strip()
    if rid.endswith("_population_guard"):
        return rid[: -len("_population_guard")]
    rid = re.sub(r"^[a-f0-9]{8}_", "", rid)
    rid = re.sub(r"^(auto|llm|tbl|path|proto)_", "", rid)
    for cid in sorted(CONDITION_BY_ID.keys(), key=len, reverse=True):
        if rid == cid or rid.startswith(cid + "_"):
            return cid
    return None


def _rule_kind_from_id(rule_id: str) -> str | None:
    rid = rule_id.lower()
    for key in sorted(_RULE_KIND_RU.keys(), key=len, reverse=True):
        if key in rid:
            return key
    return None


def rule_title_ru(rule_id: str, raw: dict | None = None) -> str:
    """Краткий русский заголовок вместо технического rule_id."""
    raw = raw or {}
    desc = str(raw.get("description_ru") or "").strip()
    if desc and len(desc) <= 100 and not desc.startswith(("http", "www")):
        return desc
    exam = raw.get("exam")
    if exam:
        return f"Обязательное обследование: {exam}"
    keyword = raw.get("keyword")
    if keyword:
        return f"Требование протокола: {keyword}"

    kind_key = _rule_kind_from_id(rule_id) or str(raw.get("rule_type") or "")
    kind_ru = _RULE_KIND_RU.get(kind_key, "Требование протокола")
    cid = extract_condition_id(rule_id)
    if cid:
        return f"{kind_ru}: {condition_title_ru(cid)}"
    return kind_ru


def decision_ru(decision: str | None) -> str:
    return _DECISION_RU.get(str(decision or "unknown"), "Не определено")


def found_status_ru(status: str | None) -> str:
    return _FOUND_STATUS_RU.get(str(status or "unknown"), "Не определено")


def rule_type_ru(rule_type: str | None) -> str:
    rt = str(rule_type or "")
    return _RULE_TYPE_RU.get(rt, _RULE_KIND_RU.get(rt, rt.replace("_", " ")))


def population_ru(value: str | None) -> str:
    v = str(value or "").lower().strip()
    return _POPULATION_RU.get(v, v)


def localize_message_ru(message: str | None) -> str:
    """Заменяет англ. термины аудитории и технические хвосты в message_ru."""
    if not message:
        return ""
    text = str(message)
    for en, ru in _POPULATION_RU.items():
        text = re.sub(rf"\b{re.escape(en)}\b", ru, text, flags=re.IGNORECASE)
    text = re.sub(
        r"\(rule_id[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Убрать технический rule_id из скобок в diagnostic_criterion
    text = re.sub(
        r"\(([a-f0-9]{6,}_[\w]+)\)",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()
