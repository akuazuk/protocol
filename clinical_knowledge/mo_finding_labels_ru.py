"""Русские названия кодов замечаний МО (deep / v3 / v4).

Если в dim_finding или fact_mo_finding лежит сам код вместо title_ru,
UI и отчёты показывают человекочитаемую формулировку.
"""
from __future__ import annotations

import re

_A_BLOCK_RU = {
    "complaints": "жалобы",
    "anamnesis": "анамнез",
    "objective_status": "объективный статус",
    "diagnosis": "диагноз",
    "exams": "рекомендации по обследованию",
    "treatment": "рекомендации по лечению",
    "follow_up": "план наблюдения",
}

FINDING_TITLE_RU: dict[str, str] = {
    "B_dx_no_support": "Диагноз не подкреплён жалобами, анамнезом или осмотром",
    "B_icd_invalid": "Код МКБ отсутствует или неверного формата",
    "B_icd_mismatch_mis": "Код МКБ в тексте не совпадает с диагнозом в МИС",
    "B_icd_dir_no_match": "Формулировка диагноза не найдена в справочнике МКБ",
    "B_icd_dir_code_unknown": "Код МКБ отсутствует в справочнике",
    "B_icd_dir_text_mismatch": "Формулировка диагноза слабо согласуется с рубрикой МКБ",
    "B_exams_gap": "Не отражены обязательные обследования протокола",
    "B_tx_gap": "Не отражено обязательное лечение протокола",
    "B_criteria_absent": "Критерии диагноза по протоколу не отражены",
    "B_tx_offprotocol": "Лечение не соответствует группам протокола",
    "C_red_flag": "Красный флаг без маршрутизации",
    "C_red_flag_unrouted": "Красный флаг без маршрутизации",
    "C_uncertainty_unrouted": "Клиническая неопределённость без маршрутизации",
    "C_nsaid_dup": "Дублирование НПВС",
    "C_ddi": "Потенциально опасное лекарственное взаимодействие",
    "C_high_alert_no_dose": "Препарат высокого риска без указания дозы",
    "C_drug_unresolved": "Препарат не удалось надёжно распознать",
    "D_reg55_p0": "Критический дефект по постановлению МЗ № 55",
    "D_reg55_gap": "Невыполненный критерий качества по постановлению МЗ № 55",
    "E_template_copy": "Подозрение на копирование шаблона между случаями",
    # Concordance shadow (E1/E3) - mo_concordance_v1
    "finding_not_in_diagnosis": "Находка в статусе не отражена в диагнозе",
    "anamnesis_thin_for_duration": "Анамнез слишком краток для длительности жалобы",
    "underworkup_chronic_red_flag": "Недостаточный объём обследования при хроническом сценарии",
    "plan_laterality_mismatch": "Латеральность плана не совпадает с жалобой",
    "icd_weakly_supported": "Код МКБ слабо поддержан клинической картиной",
    "pediatric_limp_ddx_not_addressed": "Не закрыт детский DDx длительной хромоты",
}

SEVERITY_LABEL_RU: dict[str, str] = {
    "P0": "P0 · критично",
    "P1": "P1 · клинически важно",
    "P2": "P2 · оформление",
    "P3": "P3 · формально",
}

SEVERITY_HINT_RU: dict[str, str] = {
    "P0": "Риск вреда пациенту или критический дефект качества. Ограничивает итоговую оценку.",
    "P1": "Клинический дефект: влияет на диагноз, обследование или лечение.",
    "P2": "Дефект документирования или оформления записи.",
    "P3": "Формальное замечание, без прямого клинического риска.",
}

_SOURCE_EXACT_RU: dict[str, str] = {
    "DDInter": (
        "База лекарственных взаимодействий DDInter: проверка сочетаний препаратов "
        "в рекомендациях по лечению."
    ),
    "ISMP/клин.практика": (
        "Рекомендации ISMP и клиническая практика: недопустимо одновременное "
        "назначение нескольких НПВС без обоснования."
    ),
    "ISMP high-alert": (
        "Список препаратов высокого риска ISMP: для таких средств нужна явная доза."
    ),
    "Пост. №55": (
        "Постановление МЗ Республики Беларусь от 21.05.2021 № 55 "
        "(критерии качества медицинской помощи, уровень случая)."
    ),
    "mo_concordance_v1": (
        "Автопроверка согласованности жалоб/статуса, диагноза и плана (черновик)."
    ),
    "mo_icd_directory_v1": (
        "Сверка формулировки диагноза и кода со справочником МКБ (черновик, не подбор КП)."
    ),
    "PDQI-9/№55": "Критерии полноты записи (PDQI-9) и постановление МЗ № 55.",
    "МКБ-10": "Справочник кодов МКБ-10.",
    "§3.4 chain": "Цепочка обоснования диагноза (жалобы → осмотр → диагноз).",
    "§3.4": "Методика обоснования клинических решений (§3.4).",
    "protocol.required_exams": "Обязательные обследования из клинического протокола МЗ.",
    "protocol.diagnostic_criteria": "Диагностические критерии клинического протокола МЗ.",
    "protocol.treatment": "Рекомендации по лечению из клинического протокола МЗ.",
    "protocol.red_flags": "Красные флаги клинического протокола МЗ.",
    "mis_data.diagnos": "Структурированный диагноз из МИС (mis_data).",
    "advisory:exact_jaccard_5_shingles_v1": (
        "Алгоритм сходства текста (5-shingle Jaccard): сравнение формулировок между случаями."
    ),
}

_GENERIC_DETAIL_MARKERS = (
    "требует проверки, что индивидуальные данные",
    "замечание требует проверки полноты",
    "влияет на полноту и безопасность медицинской записи",
)

_CODE_LIKE = re.compile(r"^[A-E]_[A-Za-z0-9_]+$")
_TEMPLATE_PAIR_RE = re.compile(
    r"^template_pair:([^:]+):(.+)$",
    flags=re.IGNORECASE,
)


def finding_label_ru(code: str, stored_title: str | None = None) -> str:
    """Вернуть русское название замечания.

    Если stored_title уже по-русски и не равен коду - оставляем его.
    Иначе берём каталог / шаблон A_missing_*.
    """
    cid = str(code or "").strip()
    stored = str(stored_title or "").strip()
    if stored and stored != cid and not _CODE_LIKE.match(stored):
        # Старый короткий ярлык шаблона - заменяем на каталог.
        if cid == "E_template_copy" and "шаблонност" in stored.lower():
            return FINDING_TITLE_RU[cid]
        return stored
    if cid in FINDING_TITLE_RU:
        return FINDING_TITLE_RU[cid]
    if cid.startswith("A_missing_"):
        block = cid[len("A_missing_") :]
        ru = _A_BLOCK_RU.get(block, block.replace("_", " "))
        return f"Не заполнен блок: {ru}"
    if cid.startswith("C_") and cid not in FINDING_TITLE_RU:
        return f"Замечание по безопасности: {cid[2:].replace('_', ' ')}"
    if stored:
        return stored
    return f"Замечание: {cid}" if cid else "Замечание"


def severity_label_ru(severity: str | None) -> str:
    key = str(severity or "").strip().upper()
    return SEVERITY_LABEL_RU.get(key, "Проверить" if not key else key)


def severity_hint_ru(severity: str | None) -> str:
    key = str(severity or "").strip().upper()
    return SEVERITY_HINT_RU.get(key, "Тяжесть не задана - нужна ручная оценка методиста.")


def source_ref_display_ru(source_ref: str | None) -> str:
    """Человекочитаемое описание источника замечания (всегда по-русски)."""
    raw = str(source_ref or "").strip()
    if not raw:
        return "Источник не указан - замечание сформировано правилами оценки МО."
    if raw in _SOURCE_EXACT_RU:
        return _SOURCE_EXACT_RU[raw]
    m = _TEMPLATE_PAIR_RE.match(raw)
    if m:
        pair_id, other_id = m.group(1), m.group(2)
        return (
            "Сравнение текста с другим медицинским осмотром: обнаружено высокое "
            f"текстовое сходство (пара {pair_id[:12]}…). "
            f"Сопоставленный визит/запись: {other_id}. "
            "Проверьте, не скопирован ли шаблон без индивидуализации."
        )
    if raw.startswith("advisory:"):
        return (
            "Внутреннее правило-советник системы оценки МО "
            f"({raw.split(':', 1)[-1]})."
        )
    if "№55" in raw or "55" in raw and "Пост" in raw:
        return _SOURCE_EXACT_RU["Пост. №55"]
    # Уже похоже на русское предложение - оставляем.
    if re.search(r"[А-Яа-яЁё]", raw) and " " in raw and ":" not in raw[:20]:
        return raw
    return f"Источник правила оценки: {raw}"


def is_generic_finding_detail(detail: str | None) -> bool:
    text = str(detail or "").strip().lower()
    if not text:
        return True
    return any(marker in text for marker in _GENERIC_DETAIL_MARKERS)


def enrich_finding_detail_ru(
    *,
    code: str,
    detail: str | None,
    source_ref: str | None,
    title_ru: str | None = None,
) -> str:
    """Подставить понятный detail вместо шаблонных «требует проверки…»."""
    text = str(detail or "").strip()
    if text and not is_generic_finding_detail(text):
        return text
    cid = str(code or "").strip()
    if cid == "E_template_copy":
        return source_ref_display_ru(source_ref)
    if cid == "D_reg55_p0":
        return (
            "По критериям постановления МЗ № 55 зафиксирован критический (P0) дефект. "
            "Ниже в источнике - база правила; при пустом списке критериев это может быть "
            "устаревшая оценка - перепроверьте вручную."
        )
    if cid == "D_reg55_gap":
        return text or (
            "Не выполнен один или несколько критериев качества постановления МЗ № 55 "
            "(не критический уровень). См. перечень в тексте замечания."
        )
    if cid == "C_ddi":
        return (
            text
            or "В рекомендациях по лечению найдены препараты с потенциально значимым "
            "взаимодействием. Нужна проверка дозировок, показаний и мониторинга."
        )
    if cid == "C_nsaid_dup":
        return (
            text
            or "В плане лечения одновременно указаны несколько НПВС. "
            "Оставьте один препарат или обоснуйте комбинацию."
        )
    title = finding_label_ru(cid, title_ru)
    return (
        f"{title}: автоматическое замечание по правилам оценки МО. "
        "Откройте цитату и источник, затем подтвердите или отклоните."
    )

