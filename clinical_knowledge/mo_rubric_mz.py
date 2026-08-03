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
    has_marker = _has_marker(text, markers)
    has_interval = bool(re.search(r"(через\s+\d+|\d+\s*(дн|день|недел|мес)|раз\s+в)", text.lower()))
    if has_marker and has_interval:
        return _item(
            crit,
            score=1.0,
            reason="Кратность наблюдения указана с интервалом; сверка с КП - следующий этап",
            evidence=text,
        )
    if has_marker:
        return _item(
            crit,
            score=0.5,
            reason="Упоминание наблюдения есть; интервал/сверка с КП частичные",
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


def summarize_rubric_batch(
    results: list[Mapping[str, Any]],
    *,
    specialties: list[str] | None = None,
) -> dict[str, Any]:
    """Агрегат top-fail по списку evaluate_mo_rubric_mz() результатов."""
    fail_counts: dict[str, dict[str, Any]] = {}
    scored_cases = 0
    pct_sum = 0.0
    pct_n = 0
    specialty_fail: dict[str, dict[str, dict[str, int]]] = {}
    for idx, result in enumerate(results):
        if not result or not result.get("ok"):
            continue
        scored_cases += 1
        specialty = ""
        if specialties and idx < len(specialties):
            specialty = str(specialties[idx] or "").strip() or "Без специальности"
        if isinstance(result.get("rubric_pct"), (int, float)):
            pct_sum += float(result["rubric_pct"])
            pct_n += 1
        for item in result.get("criteria") or []:
            cid = str(item.get("id") or "")
            if not cid:
                continue
            bucket = fail_counts.setdefault(
                cid,
                {
                    "id": cid,
                    "title": item.get("title") or cid,
                    "group": item.get("group"),
                    "zero_n": 0,
                    "half_n": 0,
                    "full_n": 0,
                    "na_n": 0,
                    "scored_n": 0,
                },
            )
            score = item.get("score")
            if not isinstance(score, (int, float)):
                bucket["na_n"] += 1
                continue
            bucket["scored_n"] += 1
            if score >= 0.999:
                bucket["full_n"] += 1
                level = "full"
            elif score >= 0.499:
                bucket["half_n"] += 1
                level = "half"
            else:
                bucket["zero_n"] += 1
                level = "zero"
            if specialty and level in {"zero", "half"}:
                cell = specialty_fail.setdefault(specialty, {}).setdefault(
                    cid, {"zero_n": 0, "half_n": 0, "scored_n": 0}
                )
                cell["scored_n"] += 1
                cell["zero_n" if level == "zero" else "half_n"] += 1
    top_fail = []
    for bucket in fail_counts.values():
        scored_n = int(bucket["scored_n"])
        if not scored_n:
            continue
        fail_n = int(bucket["zero_n"]) + int(bucket["half_n"])
        bucket["fail_pct"] = round(100.0 * fail_n / scored_n, 1)
        bucket["zero_pct"] = round(100.0 * int(bucket["zero_n"]) / scored_n, 1)
        top_fail.append(bucket)
    top_fail.sort(key=lambda row: (-float(row["fail_pct"]), -int(row["zero_n"]), str(row["id"])))

    specialty_rows = []
    for specialty, criteria_map in specialty_fail.items():
        worst = sorted(
            (
                {
                    "id": cid,
                    "zero_n": vals["zero_n"],
                    "half_n": vals["half_n"],
                    "fail_n": vals["zero_n"] + vals["half_n"],
                }
                for cid, vals in criteria_map.items()
            ),
            key=lambda row: (-int(row["fail_n"]), str(row["id"])),
        )
        specialty_rows.append(
            {
                "specialty": specialty,
                "fail_n": sum(int(x["fail_n"]) for x in worst),
                "top_criteria": worst[:3],
            }
        )
    specialty_rows.sort(key=lambda row: (-int(row["fail_n"]), str(row["specialty"])))

    return {
        "ok": True,
        "cases_n": scored_cases,
        "avg_rubric_pct": round(pct_sum / pct_n, 1) if pct_n else None,
        "top_fail": top_fail[:13],
        "by_specialty": specialty_rows[:12],
        "primary": False,
        "source": "config/mo_rubric_mz.yaml",
    }

def build_rubric_summary_from_sources(
    *,
    date_from: str = "",
    date_to: str = "",
    limit: int = 120,
) -> dict[str, Any]:
    """Shadow-сводка по защищённым дневным срезам (без patient_id в ответе)."""
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
            meta = {
                "visit_date": str(row.get("date") or row.get("visit_date") or day_key)[:10],
                "visit_time": str(row.get("visit_time") or row.get("time") or ""),
                "diagnosis_code": str(row.get("mkb_code_main") or row.get("diagnosis_code") or ""),
                "diagnosis_short": str(row.get("clinical_diagnosis") or row.get("mis_diagnos") or ""),
            }
            results.append(evaluate_mo_rubric_mz(clinical=clinical, meta=meta))
            specialties.append(
                str(row.get("specialization") or row.get("specialty") or row.get("doctor_specialization") or "").strip()
            )

    summary = summarize_rubric_batch(results, specialties=specialties)
    summary["date_from"] = start_d.isoformat()
    summary["date_to"] = end_d.isoformat()
    summary["sample_n"] = len(results)
    summary["files_scanned"] = len(files)
    summary["available"] = bool(results)
    if not results:
        summary["reason"] = "Нет клинических строк в защищённых срезах за период"
    return summary
