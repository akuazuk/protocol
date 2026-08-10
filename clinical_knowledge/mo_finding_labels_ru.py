"""Русские названия кодов замечаний МО (deep / v3 / v4).

Если в dim_finding или fact_mo_finding лежит сам код вместо title_ru,
UI и отчёты показывают человекочитаемую формулировку.

Уровни тяжести в UI - короткие русские слова (не «P0»/«P1»).
Приоритет очереди - по формуле оценки (см. priority_from_score).
"""
from __future__ import annotations

import re
from typing import Any

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
    "B_icd_invalid": "Код МКБ неверного формата или нет ни диагноза, ни кода",
    "B_icd_mismatch_mis": "Код МКБ в тексте не совпадает с диагнозом в МИС",
    "B_dx_absent": "Диагноз отсутствует в МО",
    "B_icd_dir_no_match": "Формулировка диагноза не найдена в справочнике МКБ",
    "B_icd_dir_code_unknown": "Код МКБ отсутствует в справочнике",
    "B_icd_dir_text_mismatch": "Формулировка диагноза слабо согласуется с рубрикой МКБ",
    "B_icd_name_no_match": "Название диагноза не сопоставлено со справочником МКБ",
    "B_icd_name_weak_match": "Формулировка диагноза слабо совпадает со справочником МКБ",
    "B_complaint_exam_mismatch": "Жалоба не согласуется с осмотром",
    "B_dx_not_in_exam": "Диагноз не отражён в осмотре",
    "B_tentative_dx_weak_support": "Предположительный диагноз слабо поддержан осмотром",
    "B_chronic_dx_therapy_absent": "Хронический диагноз без описания текущей терапии",
    "B_treatment_before_confirmed_dx": "Лечение назначено при неподтверждённом диагнозе",
    "B_complaint_not_addressed_in_plan": "Жалоба не закрыта планом",
    "A_text_noise": "В тексте МО есть опечатки или мусор OCR",
    "B_icd_llm_review_yes": "LLM: формулировка согласуется с кодом/рубрикой МКБ",
    "B_icd_llm_review_partial": "LLM: частичное согласие формулировки с МКБ",
    "B_icd_llm_review_no": "LLM: формулировка не согласуется с кодом/рубрикой МКБ",
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

# Короткие русские слова для UI (без префикса P0/P1).
# P2 = умеренный клинический сигнал; P3 = оформление/полнота записи.
SEVERITY_LABEL_RU: dict[str, str] = {
    "P0": "Критично",
    "P1": "Важно",
    "P2": "Умеренно",
    "P3": "Оформление",
}

# CSS-тон бейджа (.status.<tone>).
SEVERITY_TONE_CSS: dict[str, str] = {
    "P0": "critical",
    "P1": "important",
    "P2": "check",
    "P3": "formal",
}

SEVERITY_HINT_RU: dict[str, str] = {
    "P0": "Риск вреда пациенту или критический дефект качества. Ограничивает итоговую оценку.",
    "P1": "Клинически важно: влияет на безопасность, диагноз или лечение.",
    "P2": "Умеренный сигнал: требует внимания, обычно без немедленного риска вреда.",
    "P3": "Замечание по оформлению или полноте записи, без прямого клинического риска.",
}

_SEVERITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

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
    "mo_icd_name_match_v1": (
        "Сверка названия диагноза с формулировками справочника МКБ без учёта кодов "
        "(черновик; опечатки и неточные названия)."
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


def severity_tone_css(severity: str | None) -> str:
    key = str(severity or "").strip().upper()
    return SEVERITY_TONE_CSS.get(key, "review")


def severity_hint_ru(severity: str | None) -> str:
    key = str(severity or "").strip().upper()
    return SEVERITY_HINT_RU.get(key, "Тяжесть не задана - нужна ручная оценка методиста.")


def catalog_has_reg55_p0_criteria() -> bool:
    """Есть ли в актуальном каталоге №55 реальные score-eligible P0-критерии."""
    try:
        from clinical_knowledge.reg55_criteria import _load_reg

        reg = _load_reg() or {}
        for item in reg.get("criteria") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("severity") or "").upper() != "P0":
                continue
            if item.get("score_eligible") is False:
                continue
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def demote_stale_reg55_p0(
    *,
    code: str,
    severity: str | None,
    title_ru: str | None = None,
) -> dict[str, Any]:
    """Ложный D_reg55_p0 (каталог без P0) → P1 + формулировка gap.

    Не трогает настоящие клинические P0 (DDI и т.п.).
    """
    cid = str(code or "").strip()
    sev = str(severity or "").strip().upper()
    title = finding_label_ru(cid, title_ru)
    demoted = False
    if cid == "D_reg55_p0" and sev == "P0" and not catalog_has_reg55_p0_criteria():
        sev = "P1"
        title = FINDING_TITLE_RU.get("D_reg55_gap", title)
        demoted = True
    return {
        "code": cid,
        "severity": sev,
        "title_ru": title,
        "demoted_stale_reg55_p0": demoted,
        "severity_label_ru": severity_label_ru(sev),
        "severity_tone": severity_tone_css(sev),
        "severity_hint_ru": severity_hint_ru(sev),
    }


def priority_from_score(score: float | None) -> dict[str, Any]:
    """Приоритет очереди по формуле оценки (overall / оси), не по коду finding."""
    if score is None:
        return {
            "severity": "P1",
            "label_ru": "Проверить",
            "tone": "important",
            "score_pct": None,
        }
    try:
        value = float(score)
    except (TypeError, ValueError):
        return {
            "severity": "P1",
            "label_ru": "Проверить",
            "tone": "important",
            "score_pct": None,
        }
    if value < 40:
        sev = "P0"
    elif value < 60:
        sev = "P1"
    elif value < 75:
        sev = "P2"
    else:
        sev = "P3"
    return {
        "severity": sev,
        "label_ru": severity_label_ru(sev),
        "tone": severity_tone_css(sev),
        "score_pct": round(value, 1),
    }


def worse_severity(a: str | None, b: str | None) -> str:
    ra = _SEVERITY_RANK.get(str(a or "").upper(), 9)
    rb = _SEVERITY_RANK.get(str(b or "").upper(), 9)
    return str(a or b or "P2").upper() if ra <= rb else str(b or a or "P2").upper()


def recompute_overall_from_axes(axes: dict[str, Any] | None) -> float | None:
    """Взвешенный overall по осям (без risk-cap) - для отображения после demote P0."""
    if not isinstance(axes, dict) or not axes:
        return None
    try:
        from clinical_knowledge.kz_evaluation_engine import _AXIS_WEIGHTS
    except Exception:  # noqa: BLE001
        weights = {
            "documentation": 0.30,
            "clinical_concordance": 0.35,
            "safety": 0.25,
            "regulatory": 0.10,
        }
    else:
        weights = dict(_AXIS_WEIGHTS)
    parts: list[tuple[float, float]] = []
    for axis, weight in weights.items():
        raw = axes.get(axis)
        if isinstance(raw, (int, float)):
            parts.append((float(raw), float(weight)))
    if not parts:
        return None
    return round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 1)


def queue_priority_for_case(
    *,
    finding_severity: str | None,
    score_pct: float | None,
    axes: dict[str, Any] | None = None,
    demoted_stale_reg55_p0: bool = False,
) -> dict[str, Any]:
    """Итоговый приоритет строки очереди: худшее из (формула, тяжесть finding)."""
    display_score = score_pct
    if demoted_stale_reg55_p0:
        axis_score = recompute_overall_from_axes(axes)
        if axis_score is not None:
            display_score = axis_score
        elif isinstance(axes, dict) and isinstance(axes.get("regulatory"), (int, float)):
            # fallback: хотя бы формула №55, не застрявший cap 40%
            if display_score is None or float(display_score) <= 40.0:
                display_score = float(axes["regulatory"])
    from_score = priority_from_score(display_score)
    finding_sev = str(finding_severity or "").strip().upper() or from_score["severity"]
    # После demote №55 приоритет ведём по формуле; иначе учитываем и finding.
    if demoted_stale_reg55_p0:
        final = str(from_score["severity"])
    else:
        final = worse_severity(finding_sev, from_score["severity"])
    return {
        "severity": final,
        "label_ru": severity_label_ru(final),
        "tone": severity_tone_css(final),
        "score_pct": from_score.get("score_pct"),
        "formula_pct": display_score if isinstance(display_score, (int, float)) else None,
    }


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
        if not catalog_has_reg55_p0_criteria():
            return (
                "Устаревшая метка критического дефекта по №55: в актуальном каталоге "
                "нет критериев уровня «Критично». Смотрите средний балл по формуле №55 "
                "и перечень невыполненных пунктов - это не тикет очереди разбора."
            )
        return (
            "По критериям постановления МЗ № 55 зафиксирован критический дефект. "
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

