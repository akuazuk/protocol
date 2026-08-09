"""Оценка МО по пост. МЗ №55 прил.1 разд. V (section pack).

Пайплайн: router → pack → applicable|n/a → score 0|0.5|1 → pct (п.12) → band (п.13).
№127 используется только как evidence helper для помеченных пунктов.
Не заменяет deep A/B/C; целевая замена binary evaluate_reg55 + shadow rubric_mz.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "mo_reg55_section_packs.yaml"

_ICD_RE = re.compile(r"\b[A-TV-ZА-Яа-я][0-9]{2}(?:\.[0-9]{1,2})?\b", re.IGNORECASE)
_FOLLOW_MARKERS = ("явк", "наблюден", "контроль", "повторн", "через", "кратност", "диспансер")
_CONSENT_MARKERS = ("информированн", "добровольн", "согласи", "отказ от оказания")
_VACCINE_MARKERS = ("привив", "вакцин", "нацкалендар", "по возрасту")
_ANTHROPO_MARKERS = ("вес", "рост", "имт", "физическ")
_VITAL_MARKERS = ("а/д", "ад ", "давлен", "чсс", "пульс", "температур")


@lru_cache(maxsize=1)
def load_section_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw.get("packs"), dict) or not raw["packs"]:
        raise ValueError("mo_reg55_section_packs_missing")
    return raw


def _norm(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "-"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _clinical_blob(case: Mapping[str, Any]) -> dict[str, Any]:
    clinical = case.get("clinical") if isinstance(case.get("clinical"), dict) else {}
    out = dict(case)
    for key, val in clinical.items():
        if key not in out or not _norm(out.get(key)):
            out[key] = val
    return out


def _first_text(blob: Mapping[str, Any], fields: list[str]) -> str:
    parts = [_norm(blob.get(f)) for f in fields]
    return " ".join(p for p in parts if p).strip()


def _age_years(case: Mapping[str, Any], blob: Mapping[str, Any]) -> float | None:
    for key in ("patient_age_years", "age_years", "age"):
        raw = case.get(key, blob.get(key))
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        text = _norm(raw)
        if text:
            try:
                return float(text.replace(",", "."))
            except ValueError:
                pass
    return None


def _flag(case: Mapping[str, Any], name: str) -> bool:
    flags = case.get("flags") if isinstance(case.get("flags"), dict) else {}
    if flags.get(name) is True:
        return True
    if case.get(name) is True:
        return True
    meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
    return meta.get(name) is True


def _has_prior(case: Mapping[str, Any]) -> bool:
    if _flag(case, "has_prior_visit"):
        return True
    prior = case.get("prior_clinical") or case.get("prior_visit")
    return bool(prior)


def resolve_section_pack(case: Mapping[str, Any] | None) -> dict[str, Any]:
    """Выбрать pack_id по специальности (router YAML)."""
    cfg = load_section_config()
    blob = _clinical_blob(case or {})
    specialty = _norm(
        blob.get("doctor_specialization")
        or blob.get("specialty")
        or blob.get("specialization")
        or (case or {}).get("doctor_specialization")
        or ""
    ).lower()
    for rule in cfg.get("router") or []:
        pack_id = str(rule.get("pack_id") or "")
        markers = list(rule.get("specialty_markers") or [])
        if "*" in markers:
            pack = (cfg.get("packs") or {}).get(pack_id) or {}
            return {
                "pack_id": pack_id,
                "label_ru": pack.get("label_ru") or pack_id,
                "points_range": pack.get("points_range"),
                "provenance": pack.get("provenance"),
                "matched_specialty": specialty or None,
            }
        if any(m.lower() in specialty for m in markers if m and m != "*"):
            pack = (cfg.get("packs") or {}).get(pack_id) or {}
            return {
                "pack_id": pack_id,
                "label_ru": pack.get("label_ru") or pack_id,
                "points_range": pack.get("points_range"),
                "provenance": pack.get("provenance"),
                "matched_specialty": specialty or None,
            }
    # fallback
    pack = (cfg.get("packs") or {}).get("specialist_amb_core") or {}
    return {
        "pack_id": "specialist_amb_core",
        "label_ru": pack.get("label_ru") or "specialist_amb_core",
        "points_range": pack.get("points_range"),
        "provenance": pack.get("provenance"),
        "matched_specialty": specialty or None,
    }


def band_from_pct(pct: float | None) -> dict[str, Any]:
    cfg = load_section_config()
    bands = cfg.get("bands") or {}
    if pct is None:
        b = bands.get("unscored") or {}
        return {
            "code": "unscored",
            "label_ru": b.get("label_ru") or "Не оценено",
            "detail_ru": b.get("detail_ru") or "",
        }
    if pct >= 80.0:
        code = "compliant_min"
    elif pct >= 55.0:
        code = "compliant_measures"
    else:
        code = "noncompliant"
    b = bands.get(code) or {}
    return {
        "code": code,
        "label_ru": b.get("label_ru") or code,
        "detail_ru": b.get("detail_ru") or "",
    }


def _item(
    crit: Mapping[str, Any],
    *,
    score: float | None,
    reason: str,
    evidence: str = "",
) -> dict[str, Any]:
    if score is None:
        label = "n/a"
    elif score >= 0.999:
        label = "1"
    elif score >= 0.499:
        label = "0.5"
    else:
        label = "0"
    return {
        "point": crit.get("point"),
        "title": crit.get("title"),
        "group": crit.get("group"),
        "rule": crit.get("rule"),
        "applicable": score is not None,
        "score": score,
        "score_label": label,
        "reason": reason,
        "evidence": (evidence or "")[:240],
        "evidence_from_127": bool(crit.get("evidence_from_127")),
    }


def _score_plan_text(
    crit: Mapping[str, Any],
    blob: Mapping[str, Any],
    block_scores: Mapping[str, Any],
) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="План не указан")
    key = str(crit.get("block_score_key") or "")
    aligned = block_scores.get(key)
    aligned_f = float(aligned) if isinstance(aligned, (int, float)) else None
    if aligned_f is not None and aligned_f >= 60:
        return _item(crit, score=1.0, reason="План есть и alignment с протоколом достаточный", evidence=text)
    return _item(crit, score=0.5, reason="План указан; сверка с КП частичная/неизвестна", evidence=text)


def _score_diagnosis(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = _first_text(blob, list(crit.get("fields") or []) or ["clinical_diagnosis", "mis_diagnos"])
    if not diagnosis:
        diagnosis = _norm(blob.get("diagnosis_short"))
    icd = _norm(blob.get("mkb_code_main") or blob.get("diagnosis_code"))
    if not icd:
        m = _ICD_RE.search(diagnosis)
        icd = m.group(0) if m else ""
    # битые кириллические коды / пробелы в коде → частичный балл
    broken = bool(re.search(r"[А-Яа-я]\s*\d{2}|\b[A-TV-Z]\s+\d{2}", diagnosis))
    support = sum(
        1
        for key in ("complaints", "anamnesis_doctor", "anamnesis_auto", "objective_status")
        if _norm(blob.get(key))
    )
    allergy_conflict = (
        "аллерг" in diagnosis.lower()
        and re.search(r"аллергическ\w*\s+реакц\w*\s*:\s*нет", _first_text(blob, ["anamnesis_doctor", "anamnesis_auto"]).lower())
    )
    if diagnosis and icd and support >= 2 and not broken and not allergy_conflict:
        return _item(crit, score=1.0, reason="Диагноз с МКБ и клинической цепочкой", evidence=f"{icd} · {diagnosis}")
    if diagnosis and (icd or support >= 1):
        reason = "Диагноз частичный"
        if broken:
            reason += " (битая запись МКБ)"
        if allergy_conflict:
            reason += " (противоречие аллергоанамнеза)"
        if not icd:
            reason += " (нет МКБ)"
        return _item(crit, score=0.5, reason=reason, evidence=diagnosis)
    return _item(crit, score=0.0, reason="Диагноз и/или МКБ отсутствуют")


def _score_diagnostics(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []) or ["exam_data"])
    if not text:
        return _item(crit, score=None, reason=str(crit.get("na_reason") or "Диагностика не выполнена на визите"))
    if len(text) >= 40:
        return _item(crit, score=1.0, reason="Результаты диагностики указаны", evidence=text)
    return _item(crit, score=0.5, reason="Результаты диагностики краткие", evidence=text)


def _score_physical(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []) or ["objective_status"]).lower()
    if not text:
        return _item(crit, score=0.0, reason="Нет данных осмотра для оценки физразвития")
    hits = sum(1 for m in _ANTHROPO_MARKERS if m in text)
    if hits >= 2 and len(text) >= 80:
        return _item(crit, score=0.5, reason="Оценка на визите есть; кратность ≥1×/год не подтверждена", evidence=text)
    if hits >= 1:
        return _item(crit, score=0.5, reason="Антропометрия частичная; годичная кратность не проверена", evidence=text)
    return _item(crit, score=0.0, reason="Комплексная оценка/физразвитие не отражены", evidence=text)


def _score_vaccines(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or [])).lower()
    if not text or not any(m in text for m in _VACCINE_MARKERS):
        return _item(crit, score=0.0, reason="Сведения о прививках не найдены")
    # Полный балл только при явной сверке с календарём / перечне прививок.
    if "календар" in text or re.search(r"(акдс|корь|краснух|паротит|гепатит|бцж|полиомиелит)", text):
        return _item(crit, score=1.0, reason="Прививки указаны с детализацией/календарём", evidence=text)
    return _item(crit, score=0.5, reason="Прививки упомянуты без сверки с Нацкалендарём", evidence=text)


def _score_docs(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    fields = list(crit.get("fields") or [])
    present = sum(1 for f in fields if _norm(blob.get(f)))
    if not fields:
        return _item(crit, score=None, reason="Нет списка полей")
    ratio = present / len(fields)
    if ratio >= 0.999:
        return _item(crit, score=1.0, reason="Ключевые блоки записи заполнены")
    if ratio >= 0.5:
        return _item(crit, score=0.5, reason=f"Заполнено {present} из {len(fields)} блоков")
    return _item(crit, score=0.0, reason=f"Заполнено только {present} из {len(fields)} блоков")


def _score_emr(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []) or ["raw_text", "result"])
    if len(text) >= 200:
        return _item(crit, score=1.0, reason="Электронная запись МО доступна", evidence=text[:80])
    if text:
        return _item(crit, score=0.5, reason="Запись краткая; ЭМК не подтверждена полностью", evidence=text[:80])
    return _item(crit, score=None, reason="Нет данных о возможности ЭМК")


def _score_consent(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []) or ["raw_text", "result", "consent_text"]).lower()
    if any(m in text for m in _CONSENT_MARKERS):
        return _item(crit, score=1.0, reason="Согласие/отказ отражены в документе", evidence=text[:120])
    return _item(crit, score=0.0, reason="Согласие на вмешательства не найдено")


def _score_mo_127(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    """Структура МО по №127 как evidence для пункта №55."""
    fields = list(crit.get("fields") or [])
    present = sum(1 for f in fields if _norm(blob.get(f)))
    objective = _norm(blob.get("objective_status")).lower()
    vital_hits = sum(1 for m in _VITAL_MARKERS if m in objective)
    if present >= max(3, len(fields) - 0) and len(objective) >= 100 and vital_hits >= 1:
        return _item(
            crit,
            score=1.0,
            reason="МО оформлен в объёме, достаточном по структуре №127",
            evidence=objective[:160],
        )
    if present >= 2 and len(objective) >= 40:
        return _item(
            crit,
            score=0.5,
            reason="Структура МО частичная относительно №127",
            evidence=objective[:160],
        )
    return _item(crit, score=0.0, reason="Объём МО недостаточен относительно №127")


def _score_exams_justified(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=1.0, reason="Назначений исследований нет")
    # Явного детектора дублей нет - частичный балл при широком плане без justification markers.
    low = text.lower()
    justified = any(x in low for x in ("по показан", "согласно", "контроль", "исключить", "в связи"))
    items = len(re.findall(r"\d+\s*[\).]", text)) or len(re.findall(r"[,;]", text))
    if justified or items <= 3:
        return _item(crit, score=1.0, reason="Назначения выглядят обоснованными / немногочисленными", evidence=text)
    return _item(
        crit,
        score=0.5,
        reason="Широкий план исследований; обоснованность по КП не подтверждена текстом",
        evidence=text,
    )


def _score_follow_up(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or [])).lower()
    if not text:
        return _item(crit, score=0.0, reason="Кратность наблюдения не указана")
    has_marker = any(m in text for m in _FOLLOW_MARKERS)
    has_interval = bool(re.search(r"(через\s+\d+|\d+\s*(дн|день|недел|мес)|раз\s+в)", text))
    if has_marker and has_interval:
        return _item(crit, score=1.0, reason="Наблюдение указано с интервалом", evidence=text)
    if has_marker:
        return _item(crit, score=0.5, reason="Явка/наблюдение есть; интервал частичный", evidence=text)
    return _item(crit, score=0.0, reason="Кратность наблюдения не найдена", evidence=text)


def _score_risk_vitals(crit: Mapping[str, Any], blob: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(blob, list(crit.get("fields") or [])).lower()
    if not text:
        return _item(crit, score=0.0, reason="Нет текста для АД/факторов риска")
    vitals = sum(1 for m in _VITAL_MARKERS if m in text)
    risks = any(m in text for m in ("фактор риска", "курен", "ожирен", "гипертенз", "сахарн", "наследств"))
    if vitals >= 1 and risks:
        return _item(crit, score=1.0, reason="АД/виталс и факторы риска указаны", evidence=text)
    if vitals >= 1 or risks:
        return _item(crit, score=0.5, reason="АД/виталс или факторы риска частичные", evidence=text)
    return _item(crit, score=0.0, reason="АД и факторы риска не найдены", evidence=text)


def _eval_criterion(
    crit: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    blob: Mapping[str, Any],
    block_scores: Mapping[str, Any],
) -> dict[str, Any]:
    rule = str(crit.get("rule") or "")
    age = _age_years(case, blob)

    if rule == "na_visit_level":
        return _item(crit, score=None, reason=str(crit.get("na_reason") or "Неприменимо на уровне визита"))
    if rule == "na_unless_flag":
        flag = str(crit.get("flag") or "")
        if _flag(case, flag):
            # без отдельного scorer - частичный сигнал присутствия флага
            return _item(crit, score=0.5, reason=f"Флаг {flag} есть; детальная проверка не реализована")
        return _item(crit, score=None, reason=str(crit.get("na_reason") or f"Флаг {flag} отсутствует"))
    if rule == "na_unless_field":
        fields = list(crit.get("fields") or [])
        if any(_norm(blob.get(f)) or _norm(case.get(f)) for f in fields):
            return _item(crit, score=0.5, reason="Поле присутствует; сверка с инструкцией частичная")
        return _item(crit, score=None, reason=str(crit.get("na_reason") or "Поле отсутствует"))
    if rule == "na_unless_prior_visit":
        if not _has_prior(case):
            return _item(crit, score=None, reason=str(crit.get("na_reason") or "Нет предыдущего визита"))
        text = _first_text(blob, list(crit.get("fields") or []))
        if text:
            return _item(crit, score=1.0, reason="Есть текущий план при наличии предыдущего визита", evidence=text)
        return _item(crit, score=0.0, reason="Текущий план пуст при наличии предыдущего визита")
    if rule == "na_unless_age_lt":
        max_age = float(crit.get("max_age_years") or 0)
        if age is None or age >= max_age:
            return _item(crit, score=None, reason=str(crit.get("na_reason") or "Возраст вне критерия"))
        return _item(crit, score=0.5, reason="Возраст применим; факт проведения скрининга не подтверждён")
    if rule == "na_unless_age_gte":
        min_age = float(crit.get("min_age_years") or 0)
        if age is None or age < min_age:
            return _item(crit, score=None, reason=str(crit.get("na_reason") or "Возраст вне критерия"))
        text = _first_text(blob, ["objective_status", "exam_data", "raw_text"])
        if re.search(r"флюор|рентген|огк", text.lower()):
            return _item(crit, score=1.0, reason="Рентгенпрофилактика отражена", evidence=text)
        return _item(crit, score=0.0, reason="Рентгенпрофилактика ОГК не найдена")
    if rule == "plan_text":
        return _score_plan_text(crit, blob, block_scores)
    if rule == "diagnosis_icd":
        return _score_diagnosis(crit, blob)
    if rule == "diagnostics_completed":
        return _score_diagnostics(crit, blob)
    if rule == "physical_development":
        return _score_physical(crit, blob)
    if rule == "vaccines":
        return _score_vaccines(crit, blob)
    if rule == "docs_structured":
        return _score_docs(crit, blob)
    if rule == "emr_present":
        return _score_emr(crit, blob)
    if rule == "consent_present":
        return _score_consent(crit, blob)
    if rule == "mo_per_127":
        return _score_mo_127(crit, blob)
    if rule == "exams_justified":
        return _score_exams_justified(crit, blob)
    if rule == "follow_up":
        return _score_follow_up(crit, blob)
    if rule == "risk_vitals":
        return _score_risk_vitals(crit, blob)
    return _item(crit, score=None, reason=f"Неизвестное правило: {rule}")


def evaluate_reg55_section(case: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Оценить один случай по section pack прил.1 разд. V."""
    cfg = load_section_config()
    case = dict(case or {})
    blob = _clinical_blob(case)
    block_scores = case.get("block_scores") if isinstance(case.get("block_scores"), dict) else {}
    pack_info = resolve_section_pack(case)
    pack_id = str(pack_info["pack_id"])
    pack = (cfg.get("packs") or {}).get(pack_id) or {}
    criteria_cfg = list(pack.get("criteria") or [])

    items = [
        _eval_criterion(crit, case=case, blob=blob, block_scores=block_scores)
        for crit in criteria_cfg
    ]
    applicable = [i for i in items if isinstance(i.get("score"), (int, float))]
    score_sum = sum(float(i["score"]) for i in applicable)
    n_app = len(applicable)
    pct = round(100.0 * score_sum / n_app, 1) if n_app else None
    band = band_from_pct(pct)
    weak = [i for i in applicable if float(i["score"]) < 1.0]

    return {
        "ok": True,
        "schema_version": cfg.get("schema_version"),
        "scorer_version": cfg.get("scorer_version"),
        "regulation": cfg.get("regulation"),
        "appendix": cfg.get("appendix"),
        "section": cfg.get("section"),
        "pack_id": pack_id,
        "pack_label_ru": pack_info.get("label_ru"),
        "points_range": pack_info.get("points_range"),
        "provenance": pack_info.get("provenance"),
        "matched_specialty": pack_info.get("matched_specialty"),
        "criteria_total": len(items),
        "applicable_n": n_app,
        "na_n": len(items) - n_app,
        "score_sum": round(score_sum, 3),
        "reg55_section_pct": pct,
        "reg55_band": band["code"],
        "reg55_band_label_ru": band["label_ru"],
        "reg55_band_detail_ru": band["detail_ru"],
        "criteria": items,
        "measures": [
            {
                "point": w.get("point"),
                "title": w.get("title"),
                "score": w.get("score"),
                "reason": w.get("reason"),
            }
            for w in weak
        ],
        "source": "config/mo_reg55_section_packs.yaml",
    }


def _criterion_ui_row(item: Mapping[str, Any]) -> dict[str, Any]:
    score = item.get("score")
    if score is None:
        verdict, verdict_ru, tone_score = "na", "Не применим", None
    elif float(score) >= 0.999:
        verdict, verdict_ru, tone_score = "pass", "Выполнен", 1.0
    elif float(score) >= 0.499:
        verdict, verdict_ru, tone_score = "partial", "Частично", 0.5
    else:
        verdict, verdict_ru, tone_score = "fail", "Не выполнен", 0.0
    return {
        "id": item.get("point"),
        "point": item.get("point"),
        "point_no": item.get("point"),
        "title": item.get("title"),
        "group": item.get("group"),
        "verdict": verdict,
        "verdict_ru": verdict_ru,
        "score": tone_score,
        "score_label": item.get("score_label"),
        "applicable": item.get("applicable"),
        "whats_wrong_ru": item.get("reason") if verdict in {"fail", "partial"} else "",
        "how_checked_ru": item.get("reason") or "",
        "evidence": item.get("evidence") or "",
        "evidence_from_127": bool(item.get("evidence_from_127")),
    }


def warehouse_reg55_columns(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """Плоские колонки №55 section-pack для fact_mo_case."""
    if not isinstance(section, Mapping):
        return {
            "reg55_section_pct": None,
            "reg55_band": "unscored",
            "reg55_pack": None,
            "reg55_applicable_n": None,
            "reg55_weak_points_json": "[]",
        }
    if section.get("measures") is not None:
        weak = [
            str(m.get("point") or "").strip()
            for m in (section.get("measures") or [])
            if str(m.get("point") or "").strip()
        ]
    else:
        weak = [
            str(item.get("point") or "").strip()
            for item in (section.get("criteria") or [])
            if str(item.get("point") or "").strip()
            and isinstance(item.get("score"), (int, float))
            and float(item["score"]) < 1.0
        ]
    pct = section.get("reg55_section_pct")
    return {
        "reg55_section_pct": float(pct) if isinstance(pct, (int, float)) else None,
        "reg55_band": str(section.get("reg55_band") or "unscored"),
        "reg55_pack": str(section.get("pack_id") or "") or None,
        "reg55_applicable_n": (
            int(section["applicable_n"])
            if isinstance(section.get("applicable_n"), (int, float))
            else None
        ),
        "reg55_weak_points_json": json.dumps(weak, ensure_ascii=False),
    }


def to_reg55_detail_payload(section: Mapping[str, Any]) -> dict[str, Any]:
    """Форма для case-detail / UI (совместима с renderReg55 + band)."""
    pct = section.get("reg55_section_pct")
    criteria = [_criterion_ui_row(c) for c in (section.get("criteria") or [])]
    applicable = [c for c in criteria if c.get("verdict") != "na"]
    passed = sum(1 for c in applicable if c.get("verdict") == "pass")
    return {
        "engine": "mo_reg55_section",
        "ok": bool(section.get("ok")),
        "regulatory_compliance_pct": pct,
        "reg55_section_pct": pct,
        "reg55_band": section.get("reg55_band"),
        "reg55_band_label_ru": section.get("reg55_band_label_ru"),
        "reg55_band_detail_ru": section.get("reg55_band_detail_ru"),
        "pack_id": section.get("pack_id"),
        "pack_label_ru": section.get("pack_label_ru"),
        "points_range": section.get("points_range"),
        "provenance": section.get("provenance"),
        "passed": passed,
        "total": len(applicable),
        "applicable": len(applicable),
        "na": int(section.get("na_n") or 0),
        "failed": [
            {
                "id": m.get("point"),
                "title": m.get("title"),
                "point": m.get("point"),
                "severity": "P2",
                "score": m.get("score"),
                "how_checked_ru": m.get("reason"),
            }
            for m in (section.get("measures") or [])
        ],
        "critical_failed": [],
        "has_p0_defect": False,
        "criteria": criteria,
        "measures": list(section.get("measures") or []),
        "formula_ru": (
            "Средний балл №55 = 100 × (сумма 0/0.5/1) / (применимые пункты разд. V; "
            "n/a вне знаменателя). Градация: 80-100 / 55-79,9 / ≤54,9."
        ),
        "note_ru": (
            f"{section.get('pack_label_ru') or 'Раздел V'}. "
            f"{section.get('reg55_band_detail_ru') or ''}"
        ).strip(),
        "scorer_version": section.get("scorer_version"),
        "source": section.get("source"),
    }


def attach_reg55_section_to_detail(
    detail: dict[str, Any] | None,
    *,
    clinical: Mapping[str, Any] | None = None,
    block_scores: Mapping[str, Any] | None = None,
    live_case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Встроить section-оценку №55 в case-detail (primary)."""
    out = detail if isinstance(detail, dict) else {"ok": False}
    record = dict(out.get("record") or {})
    case: dict[str, Any] = {}
    if isinstance(live_case, dict):
        case.update(live_case)
    case.update(record)
    if isinstance(clinical, dict):
        case["clinical"] = clinical
        for key, val in clinical.items():
            if key not in case or not _norm(case.get(key)):
                case[key] = val
    if isinstance(block_scores, dict) and block_scores:
        case["block_scores"] = block_scores
    else:
        case.setdefault("block_scores", {})
    case.setdefault(
        "doctor_specialization",
        record.get("specialization")
        or record.get("specialty")
        or case.get("doctor_specialization")
        or "",
    )
    kind = str(record.get("document_kind") or case.get("document_kind") or "").strip()
    if kind == "consultation":
        kind = "clinical_visit"
    if kind and kind != "clinical_visit":
        axes = dict(out.get("axes") or {})
        out["record"] = record
        out["axes"] = axes
        out["reg55"] = {
            "engine": "mo_reg55_section",
            "regulatory_compliance_pct": None,
            "reg55_section_pct": None,
            "reg55_band": "unscored",
            "criteria": [],
            "failed": [],
            "note_ru": (
                f"Тип документа «{kind}» не оценивается по постановлению № 55 "
                "(нужен clinical_visit)."
            ),
        }
        out["reg55_section"] = out["reg55"]
        return out

    section = evaluate_reg55_section(case)
    payload = to_reg55_detail_payload(section)
    axes = dict(out.get("axes") or {})
    pct = payload.get("regulatory_compliance_pct")
    if isinstance(pct, (int, float)):
        record["reg55_pct"] = float(pct)
        record["reg55_section_pct"] = float(pct)
        record["reg55_band"] = payload.get("reg55_band")
        record["reg55_pack"] = payload.get("pack_id")
        axes["regulatory"] = float(pct)
    out["record"] = record
    out["axes"] = axes
    out["reg55"] = payload
    out["reg55_section"] = payload
    return out


def summarize_reg55_section_batch(
    results: list[Mapping[str, Any]],
    *,
    specialties: list[str] | None = None,
) -> dict[str, Any]:
    """Агрегат для Обзора: avg %, band_share, top-fail пунктов."""
    scored = [r for r in results if isinstance(r.get("reg55_section_pct"), (int, float))]
    band_counts = {
        "compliant_min": 0,
        "compliant_measures": 0,
        "noncompliant": 0,
        "unscored": 0,
    }
    for r in results:
        code = str(r.get("reg55_band") or "unscored")
        if code not in band_counts:
            code = "unscored"
        band_counts[code] += 1
    n_scored = len(scored)
    avg_pct = (
        round(sum(float(r["reg55_section_pct"]) for r in scored) / n_scored, 1)
        if n_scored
        else None
    )
    fail_counts: dict[str, dict[str, Any]] = {}
    for r in results:
        for item in r.get("criteria") or []:
            if not isinstance(item.get("score"), (int, float)):
                continue
            point = str(item.get("point") or "")
            if not point:
                continue
            bucket = fail_counts.setdefault(
                point,
                {
                    "id": point,
                    "point": point,
                    "title": item.get("title") or point,
                    "zero_n": 0,
                    "half_n": 0,
                    "fail_n": 0,
                    "applicable_n": 0,
                },
            )
            bucket["applicable_n"] += 1
            score = float(item["score"])
            if score < 0.001:
                bucket["zero_n"] += 1
                bucket["fail_n"] += 1
            elif score < 0.999:
                bucket["half_n"] += 1
                bucket["fail_n"] += 1
    top_fail = []
    for bucket in fail_counts.values():
        app_n = int(bucket["applicable_n"] or 0)
        fail_n = int(bucket["fail_n"] or 0)
        bucket["fail_pct"] = round(100.0 * fail_n / app_n, 1) if app_n else 0.0
        top_fail.append(bucket)
    top_fail.sort(key=lambda x: (-float(x["fail_pct"]), -int(x["fail_n"]), str(x["point"])))

    by_specialty: list[dict[str, Any]] = []
    if specialties and len(specialties) == len(results):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for spec, res in zip(specialties, results):
            key = (spec or "не указана").strip() or "не указана"
            grouped.setdefault(key, []).append(res)
        for spec, rows in grouped.items():
            fail_n = 0
            point_fail: dict[str, int] = {}
            for res in rows:
                for item in res.get("criteria") or []:
                    if isinstance(item.get("score"), (int, float)) and float(item["score"]) < 1.0:
                        fail_n += 1
                        pid = str(item.get("point") or "")
                        if pid:
                            point_fail[pid] = point_fail.get(pid, 0) + 1
            top_criteria = [
                {"id": pid, "fail_n": n}
                for pid, n in sorted(point_fail.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            ]
            by_specialty.append({"specialty": spec, "fail_n": fail_n, "top_criteria": top_criteria})
        by_specialty.sort(key=lambda x: (-int(x["fail_n"]), str(x["specialty"])))

    n_banded = (
        band_counts["compliant_min"]
        + band_counts["compliant_measures"]
        + band_counts["noncompliant"]
    )
    band_share = {}
    for code, n in band_counts.items():
        if code == "unscored":
            denom = len(results) or 1
        else:
            denom = n_banded or 1
        band_share[code] = {
            "n": n,
            "pct": round(100.0 * n / denom, 1) if results else 0.0,
        }
    return {
        "available": bool(results),
        "engine": "mo_reg55_section",
        "avg_reg55_section_pct": avg_pct,
        "avg_pct": avg_pct,
        "value": avg_pct,
        "sample_n": len(results),
        "scored_n": n_scored,
        "band_share": band_share,
        "top_fail": top_fail[:12],
        "by_specialty": by_specialty[:12],
        "reg55_band": band_from_pct(avg_pct)["code"] if avg_pct is not None else "unscored",
        "reg55_band_label_ru": band_from_pct(avg_pct)["label_ru"],
    }


def build_reg55_section_summary_from_sources(
    *,
    date_from: str = "",
    date_to: str = "",
    limit: int = 120,
) -> dict[str, Any]:
    """Сводка №55 section-pack по secure_cases/raw за период (как rubric-summary)."""
    from datetime import date, timedelta

    from clinical_knowledge.mo_case_document import (
        _medical_exam_roots,
        _read_source_records,
        clinical_fields_from_row,
    )

    limit = max(10, min(int(limit or 120), 300))
    end = (date_to or "")[:10]
    start = (date_from or "")[:10]
    try:
        end_d = date.fromisoformat(end) if end else date.today()
    except ValueError:
        end_d = date.today()
    try:
        start_d = date.fromisoformat(start) if start else end_d - timedelta(days=7)
    except ValueError:
        start_d = end_d - timedelta(days=7)
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    files: list[tuple[str, Path]] = []
    day = end_d
    while day >= start_d:
        key = day.isoformat()
        year, month = key[:4], key[5:7]
        for root in _medical_exam_roots():
            for path in (
                root / "secure_cases" / year / month / f"mo_{key}.csv",
                root / "raw" / year / month / f"mo_{key}.parquet",
            ):
                if path.is_file():
                    files.append((key, path))
        day -= timedelta(days=1)

    results: list[dict[str, Any]] = []
    specialties: list[str] = []
    seen_ids: set[str] = set()
    for day_key, path in files:
        if len(results) >= limit:
            break
        try:
            rows = _read_source_records(path)
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            if len(results) >= limit:
                break
            clinical = clinical_fields_from_row(row)
            if not clinical:
                continue
            rid = str(row.get("id") or row.get("mis_id") or row.get("visit_id") or "").strip()
            if rid and rid in seen_ids:
                continue
            if rid:
                seen_ids.add(rid)
            kind = str(row.get("document_kind") or "").strip()
            if kind in {"certificate", "diagnostic", "non_clinical", "empty"}:
                continue
            if kind == "consultation":
                kind = "clinical_visit"
            if kind and kind != "clinical_visit":
                continue
            case = {
                **clinical,
                "document_kind": "clinical_visit",
                "doctor_specialization": str(
                    row.get("specialization")
                    or row.get("specialty")
                    or row.get("doctor_specialization")
                    or ""
                ).strip(),
                "mkb_code_main": str(row.get("mkb_code_main") or row.get("diagnosis_code") or ""),
                "diagnosis_code": str(row.get("diagnosis_code") or row.get("mkb_code_main") or ""),
                "visit_date": str(row.get("date") or row.get("visit_date") or day_key)[:10],
                "raw_text": " ".join(
                    str(clinical.get(k) or "")
                    for k in (
                        "complaints",
                        "anamnesis_doctor",
                        "objective_status",
                        "clinical_diagnosis",
                        "exam_recommendations",
                        "treatment_recommendations",
                    )
                ),
                "block_scores": {},
            }
            results.append(evaluate_reg55_section(case))
            specialties.append(str(case.get("doctor_specialization") or "").strip())

    summary = summarize_reg55_section_batch(results, specialties=specialties)
    summary["date_from"] = start_d.isoformat()
    summary["date_to"] = end_d.isoformat()
    summary["files_scanned"] = len(files)
    if not results:
        summary["reason"] = "Нет clinical_visit в защищённых срезах за период"
        summary["available"] = False
    return summary
