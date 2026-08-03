"""Shadow-оценка МО по рубрике МЗ «Как оценивать» (0 / 0.5 / 1 / n/a).

Не заменяет deep/v4. Читает config/mo_rubric_mz.yaml и эвристики presence/depth
по клиническим полям case detail / document. Dynamics без предыдущего визита = n/a.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "mo_rubric_mz.yaml"

_ICD_RE = re.compile(r"\b[A-TV-ZА-Яа-я][0-9]{2}(?:\.[0-9]{1,2})?\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.\-][0-5]\d\b")
_DATE_RE = re.compile(r"\b(20\d{2}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]20\d{2})\b")


@lru_cache(maxsize=1)
def load_rubric_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    criteria = raw.get("criteria") or []
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("mo_rubric_mz_criteria_missing")
    ids = [str(c.get("id") or "") for c in criteria]
    if len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError("mo_rubric_mz_criteria_ids_invalid")
    return raw


def _norm_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "-"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _first_text(clinical: Mapping[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for key in fields:
        text = _norm_text(clinical.get(key))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _has_marker(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in markers if m)


def _score_label(score: float | None) -> str:
    if score is None:
        return "n/a"
    if score >= 0.999:
        return "1"
    if score >= 0.499:
        return "0.5"
    return "0"


def _item(
    crit: Mapping[str, Any],
    *,
    score: float | None,
    reason: str,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "id": crit.get("id"),
        "title": crit.get("title"),
        "group": crit.get("group"),
        "how_to_evaluate": crit.get("how_to_evaluate"),
        "regulation": crit.get("regulation"),
        "scored_by_55": bool(crit.get("scored_by_55")),
        "score": score,
        "score_label": _score_label(score),
        "reason": reason,
        "evidence": (evidence or "")[:240],
    }


def _score_core_fields(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    fields = list(crit.get("fields") or [])
    present = sum(1 for f in fields if _norm_text(clinical.get(f)))
    if not fields:
        return _item(crit, score=None, reason="Нет списка полей")
    ratio = present / len(fields)
    if ratio >= 0.999:
        score = 1.0
        reason = "Основные блоки МО заполнены"
    elif ratio >= 0.5:
        score = 0.5
        reason = f"Заполнено {present} из {len(fields)} ключевых блоков"
    else:
        score = 0.0
        reason = f"Заполнено только {present} из {len(fields)} ключевых блоков"
    return _item(crit, score=score, reason=reason)


def _score_datetime(crit: Mapping[str, Any], meta: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    date_raw = _norm_text(meta.get("visit_date") or meta.get("date") or clinical.get("visit_date"))
    time_raw = _norm_text(meta.get("visit_time") or clinical.get("visit_time"))
    blob = " ".join(x for x in (date_raw, time_raw, _norm_text(meta.get("datetime"))) if x)
    has_date = bool(date_raw) or bool(_DATE_RE.search(blob))
    has_time = bool(time_raw) or bool(_TIME_RE.search(blob))
    if has_date and has_time:
        return _item(crit, score=1.0, reason="Дата и время указаны", evidence=blob[:80])
    if has_date:
        return _item(crit, score=0.5, reason="Есть дата, время не найдено", evidence=date_raw[:80])
    return _item(crit, score=0.0, reason="Дата и время не указаны")


def _score_text_detailed(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="Поле пустое")
    markers = list(crit.get("detail_markers") or [])
    marker_hits = sum(1 for m in markers if m.lower() in text.lower())
    if len(text) >= 80 and marker_hits >= 2:
        return _item(crit, score=1.0, reason="Текст указан и детализирован", evidence=text)
    if len(text) >= 20 or marker_hits >= 1:
        return _item(crit, score=0.5, reason="Текст есть, детализация частичная", evidence=text)
    return _item(crit, score=0.0, reason="Текст слишком краткий / шаблонный", evidence=text)


def _score_text_sufficient(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="Анамнез не заполнен")
    partial = int(crit.get("min_chars_partial") or 40)
    full = int(crit.get("min_chars_full") or 120)
    if len(text) >= full:
        return _item(crit, score=1.0, reason="Объём достаточен для уточнения диагноза", evidence=text)
    if len(text) >= partial:
        return _item(crit, score=0.5, reason="Объём частичный", evidence=text)
    return _item(crit, score=0.0, reason="Объём недостаточен", evidence=text)


def _score_risk_factors(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="Нет текста для поиска факторов риска")
    markers = list(crit.get("markers") or [])
    if _has_marker(text, markers):
        return _item(crit, score=1.0, reason="Факторы риска упомянуты", evidence=text)
    # Без выделенного слота нельзя уверенно ставить 0 - частичный сигнал.
    if len(text) >= 120:
        return _item(crit, score=0.5, reason="Явных маркеров факторов риска нет", evidence=text)
    return _item(crit, score=0.0, reason="Факторы риска не указаны", evidence=text)


def _score_objective(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="Объективный статус пуст")
    markers = list(crit.get("markers") or [])
    hits = sum(1 for m in markers if m.lower() in text.lower())
    if len(text) >= 100 and hits >= 2:
        return _item(crit, score=1.0, reason="Осмотр описан в достаточном объёме", evidence=text)
    if len(text) >= 30 or hits >= 1:
        return _item(crit, score=0.5, reason="Осмотр частичный", evidence=text)
    return _item(crit, score=0.0, reason="Осмотр слишком краткий", evidence=text)


def _score_exam_data(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        # По методике поле опционально, если пациент ничего не принёс.
        return _item(crit, score=None, reason="Нет данных обследований в МО (допустимо, если не приносились)")
    if len(text) >= 40:
        return _item(crit, score=1.0, reason="Данные обследований указаны", evidence=text)
    return _item(crit, score=0.5, reason="Данные обследований краткие", evidence=text)


def _score_diagnosis(
    crit: Mapping[str, Any],
    clinical: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    diagnosis = _first_text(clinical, ["clinical_diagnosis", "mis_diagnos"])
    if not diagnosis:
        diagnosis = _norm_text(meta.get("diagnosis_short") or meta.get("diagnosis_label"))
    icd = _norm_text(meta.get("diagnosis_code") or meta.get("mkb_code_main"))
    if not icd:
        icd_match = _ICD_RE.search(diagnosis)
        icd = icd_match.group(0) if icd_match else ""
    support = sum(
        1
        for key in ("complaints", "anamnesis_doctor", "objective_status", "exam_data")
        if _norm_text(clinical.get(key))
    )
    if diagnosis and icd and support >= 2:
        return _item(
            crit,
            score=1.0,
            reason="Диагноз с МКБ и клинической цепочкой",
            evidence=f"{icd} · {diagnosis}",
        )
    if diagnosis and (icd or support >= 1):
        return _item(
            crit,
            score=0.5,
            reason="Диагноз частичный (нет МКБ или слабая цепочка)",
            evidence=diagnosis,
        )
    return _item(crit, score=0.0, reason="Диагноз и/или МКБ отсутствуют")


def _score_plan(
    crit: Mapping[str, Any],
    clinical: Mapping[str, Any],
    block_scores: Mapping[str, Any],
) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    block_key = str(crit.get("block_score_key") or "")
    aligned = block_scores.get(block_key)
    aligned_f = float(aligned) if isinstance(aligned, (int, float)) else None
    if text and aligned_f is not None and aligned_f >= 60:
        return _item(crit, score=1.0, reason="План есть и согласован с протоколом", evidence=text)
    if text and (aligned_f is None or aligned_f >= 40):
        return _item(crit, score=0.5, reason="План указан, согласование частичное/неизвестно", evidence=text)
    if text:
        return _item(crit, score=0.5, reason="План указан без подтверждённого alignment", evidence=text)
    if aligned_f is not None and aligned_f >= 60:
        return _item(crit, score=0.5, reason="Alignment есть, текст плана не найден")
    return _item(crit, score=0.0, reason="План не указан")


def _score_dynamics(
    crit: Mapping[str, Any],
    clinical: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not prior:
        return _item(crit, score=None, reason="Нет предыдущего визита - коррекция по динамике не оценивается")
    current = _first_text(clinical, list(crit.get("fields") or []))
    previous = _first_text(prior, list(crit.get("fields") or []))
    if not current:
        return _item(crit, score=0.0, reason="Текущий план пуст при наличии предыдущего визита")
    if previous and current != previous:
        return _item(crit, score=1.0, reason="План изменён относительно предыдущего визита", evidence=current)
    if previous and current == previous:
        return _item(crit, score=0.5, reason="План не изменён - коррекция не подтверждена", evidence=current)
    return _item(crit, score=0.5, reason="Предыдущий план недоступен для сравнения", evidence=current)


def _score_follow_up(crit: Mapping[str, Any], clinical: Mapping[str, Any]) -> dict[str, Any]:
    text = _first_text(clinical, list(crit.get("fields") or []))
    if not text:
        return _item(crit, score=0.0, reason="Кратность наблюдения не указана")
    markers = list(crit.get("markers") or [])
    if _has_marker(text, markers):
        return _item(
            crit,
            score=0.5,
            reason="Упоминание наблюдения есть; сверка с КП/№127 ещё не подключена",
            evidence=text,
        )
    return _item(crit, score=0.0, reason="Явной кратности наблюдения не найдено", evidence=text)


def evaluate_mo_rubric_mz(
    *,
    clinical: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    block_scores: Mapping[str, Any] | None = None,
    prior_clinical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Оценить один случай по рубрике МЗ.

    Возвращает список критериев, rubric_pct и provenance. Shadow: primary=false.
    """
    cfg = load_rubric_config()
    clinical = dict(clinical or {})
    meta = dict(meta or {})
    block_scores = dict(block_scores or {})
    prior = dict(prior_clinical) if prior_clinical else None

    items: list[dict[str, Any]] = []
    for crit in cfg.get("criteria") or []:
        rule = str(crit.get("rule") or "")
        if rule == "core_fields_completeness":
            items.append(_score_core_fields(crit, clinical))
        elif rule == "datetime_present":
            items.append(_score_datetime(crit, meta, clinical))
        elif rule == "text_detailed":
            items.append(_score_text_detailed(crit, clinical))
        elif rule == "text_sufficient":
            items.append(_score_text_sufficient(crit, clinical))
        elif rule == "risk_factors_present":
            items.append(_score_risk_factors(crit, clinical))
        elif rule == "objective_complete":
            items.append(_score_objective(crit, clinical))
        elif rule == "exam_data_optional":
            items.append(_score_exam_data(crit, clinical))
        elif rule == "diagnosis_chain_icd":
            items.append(_score_diagnosis(crit, clinical, meta))
        elif rule == "plan_present_or_aligned":
            items.append(_score_plan(crit, clinical, block_scores))
        elif rule == "dynamics_correction":
            items.append(_score_dynamics(crit, clinical, prior))
        elif rule == "follow_up_present":
            items.append(_score_follow_up(crit, clinical))
        else:
            items.append(_item(crit, score=None, reason=f"Неизвестное правило: {rule}"))

    scored = [i for i in items if isinstance(i.get("score"), (int, float))]
    rubric_pct = (
        round(100.0 * sum(float(i["score"]) for i in scored) / len(scored), 1)
        if scored
        else None
    )
    by_group: dict[str, dict[str, Any]] = {}
    groups = cfg.get("groups") or {}
    for item in items:
        group = str(item.get("group") or "other")
        bucket = by_group.setdefault(
            group,
            {"label": groups.get(group) or group, "scored_n": 0, "sum": 0.0, "na_n": 0},
        )
        if isinstance(item.get("score"), (int, float)):
            bucket["scored_n"] += 1
            bucket["sum"] += float(item["score"])
        else:
            bucket["na_n"] += 1
    for bucket in by_group.values():
        bucket["pct"] = (
            round(100.0 * bucket["sum"] / bucket["scored_n"], 1) if bucket["scored_n"] else None
        )
        bucket.pop("sum", None)

    return {
        "ok": True,
        "schema_version": cfg.get("schema_version"),
        "scorer_version": cfg.get("scorer_version"),
        "primary": bool(cfg.get("primary")),
        "rubric_pct": rubric_pct,
        "scored_n": len(scored),
        "na_n": len(items) - len(scored),
        "criteria": items,
        "by_group": by_group,
        "source": "config/mo_rubric_mz.yaml",
    }
