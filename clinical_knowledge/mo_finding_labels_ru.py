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
    "D_reg55_p0": "Критический дефект по постановлению №55",
    "E_template_copy": "Подозрение на копирование шаблона",
    # Concordance shadow (E1/E3) - mo_concordance_v1
    "finding_not_in_diagnosis": "Находка в статусе не отражена в диагнозе",
    "anamnesis_thin_for_duration": "Анамнез слишком краток для длительности жалобы",
    "underworkup_chronic_red_flag": "Недостаточный объём обследования при хроническом сценарии",
    "plan_laterality_mismatch": "Латеральность плана не совпадает с жалобой",
    "icd_weakly_supported": "Код МКБ слабо поддержан клинической картиной",
    "pediatric_limp_ddx_not_addressed": "Не закрыт детский DDx длительной хромоты",
}

_CODE_LIKE = re.compile(r"^[A-E]_[A-Za-z0-9_]+$")


def finding_label_ru(code: str, stored_title: str | None = None) -> str:
    """Вернуть русское название замечания.

    Если stored_title уже по-русски и не равен коду - оставляем его.
    Иначе берём каталог / шаблон A_missing_*.
    """
    cid = str(code or "").strip()
    stored = str(stored_title or "").strip()
    if stored and stored != cid and not _CODE_LIKE.match(stored):
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
