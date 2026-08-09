"""Оценка МО по пост. МЗ №55 прил.1 разд. V (section pack).

Пайплайн: router → pack → applicable|n/a → score 0|0.5|1 → pct (п.12) → band (п.13).
№127 используется только как evidence helper для помеченных пунктов.
Не заменяет deep A/B/C; целевая замена binary evaluate_reg55 + shadow rubric_mz.
"""
from __future__ import annotations

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
