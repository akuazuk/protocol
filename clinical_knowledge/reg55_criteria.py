"""Оценка КЗ по критериям Постановления МЗ РБ 21.05.2021 № 55 (case-level).

Детерминированно (без LLM). Считает долю выполненных критериев уровня случая
и список невыполненных со ссылкой на пункт постановления. Работает по полям
`case` из cases.jsonl (`fields_present`, `block_scores`, `diagnosis_short`,
`status`), поэтому доступно и при rebuild summary без доступа к БД.

stdlib-only: модуль импортируется батч-скриптом без Pydantic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REG_PATH = ROOT / "data" / "regulations" / "mz_2021_55.json"

@lru_cache(maxsize=1)
def _load_reg() -> dict[str, Any]:
    try:
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _nonempty(*values: Any) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if str(value).strip():
            return True
    return False


def fields_present_from_case(case: dict[str, Any] | None) -> dict[str, bool]:
    """Вывести заполненность разделов МО из case / clinical / fields_present."""
    raw = case if isinstance(case, dict) else {}
    existing = raw.get("fields_present")
    if isinstance(existing, dict) and existing:
        # нормализуем к bool; недостающие ключи добираем из текста
        base = {str(k): bool(v) for k, v in existing.items()}
    else:
        base = {}
    clinical = raw.get("clinical") if isinstance(raw.get("clinical"), dict) else {}

    def _pick(*keys: str) -> bool:
        for key in keys:
            if _nonempty(raw.get(key)) or _nonempty(clinical.get(key)):
                return True
        return False

    out = {
        "complaints": bool(base.get("complaints")) or _pick("complaints"),
        "anamnesis": bool(base.get("anamnesis"))
        or _pick("anamnesis", "anamnesis_doctor", "anamnesis_auto"),
        "objective_status": bool(base.get("objective_status")) or _pick("objective_status"),
        "exams": bool(base.get("exams"))
        or _pick("exam_recommendations", "exam_data", "exams"),
        "treatment": bool(base.get("treatment"))
        or _pick("treatment_recommendations", "treatment"),
        "diagnosis": bool(base.get("diagnosis"))
        or _pick(
            "clinical_diagnosis",
            "diagnosis_main_text",
            "diagnosis_short",
            "diagnosis_list",
            "diagnosis",
        ),
        "follow_up": bool(base.get("follow_up"))
        or _pick("dispensary_info", "return_date", "follow_up"),
    }
    return out


def prepare_case_for_reg55(case: dict[str, Any] | None) -> dict[str, Any]:
    """Подготовить case для evaluate_reg55 (fields_present + clinical keys)."""
    raw = dict(case or {})
    clinical = raw.get("clinical") if isinstance(raw.get("clinical"), dict) else {}
    for key in (
        "complaints",
        "anamnesis_doctor",
        "anamnesis_auto",
        "objective_status",
        "exam_recommendations",
        "exam_data",
        "treatment_recommendations",
        "clinical_diagnosis",
        "diagnosis_main_text",
        "diagnosis_short",
        "dispensary_info",
        "return_date",
        "mkb_code_main",
        "mis_diagnos",
    ):
        if not _nonempty(raw.get(key)) and _nonempty(clinical.get(key)):
            raw[key] = clinical.get(key)
    raw["fields_present"] = fields_present_from_case(raw)
    return raw


def _fields_present(case: dict) -> dict:
    fp = case.get("fields_present")
    if isinstance(fp, dict) and fp:
        return fp
    return fields_present_from_case(case)


def _block_score(case: dict, block: str) -> float | None:
    bs = case.get("block_scores")
    if not isinstance(bs, dict):
        return None
    val = bs.get(block)
    return float(val) if isinstance(val, (int, float)) else None


def _icd10_present(case: dict) -> bool:
    # Код по всему МО или осмысленный текст диагноза (отсутствие кода при Dx - не fail).
    from .mo_icd_resolve import assess_icd_code_requirement

    return bool(assess_icd_code_requirement(case).get("ok"))


def _diagnosis_substantiated(case: dict) -> bool:
    fp = _fields_present(case)
    return bool(
        fp.get("diagnosis")
        and fp.get("complaints")
        and (fp.get("anamnesis") or fp.get("objective_status"))
    )


def _eval_criterion(crit: dict, case: dict, thresholds: dict) -> str:
    """Возвращает 'pass' | 'fail' | 'na'.

    Пункты с ``score_eligible: false`` (служебные, не из постановления) всегда ``na``
    для формулы среднего балла; в таблице критериев они видны отдельно.
    """
    if crit.get("score_eligible") is False:
        return "na"
    check = crit.get("check")
    if check == "field_present":
        return "pass" if _fields_present(case).get(crit.get("field")) else "fail"
    if check == "icd10_present":
        return "pass" if _icd10_present(case) else "fail"
    if check == "diagnosis_substantiated":
        return "pass" if _diagnosis_substantiated(case) else "fail"
    if check == "alignment_min":
        score = _block_score(case, str(crit.get("block")))
        if score is None:
            return "na"  # блок протокола не применялся - не в знаменателе
        thr = float(thresholds.get(crit.get("threshold_key"), 50))
        return "pass" if score >= thr else "fail"
    if check == "no_manual_review":
        return "fail" if str(case.get("status")) == "manual_review_required" else "pass"
    return "na"


def _whats_wrong_ru(crit: dict, verdict: str, case: dict) -> str:
    """Кратко: что не так при fail; пусто для pass/na."""
    if verdict != "fail":
        return ""
    custom = str(crit.get("fail_ru") or "").strip()
    if custom:
        return custom
    check = crit.get("check")
    if check == "field_present":
        return f"Не заполнено поле «{crit.get('field') or 'раздел'}»."
    if check == "diagnosis_substantiated":
        return "Не хватает связки жалобы + (анамнез или осмотр) + диагноз."
    if check == "alignment_min":
        score = _block_score(case, str(crit.get("block")))
        return f"Соответствие блока «{crit.get('block')}» протоколу: {score}."
    if check == "no_manual_review":
        return "Случай в очереди ручной проверки модели."
    return "Критерий не выполнен."


def _how_checked_ru(crit: dict) -> str:
    check = crit.get("check")
    if check == "field_present":
        field = crit.get("field") or "поле"
        return f"Проверяется наличие заполненного поля «{field}» в МО/КЗ."
    if check == "icd10_present":
        return (
            "Код МКБ-10 ищется по всему тексту МО/КЗ. Если кода нет, но есть "
            "формулировка клинического диагноза - критерий не считается нарушением."
        )
    if check == "diagnosis_substantiated":
        return (
            "Диагноз считается обоснованным, если есть диагноз и жалобы, "
            "плюс анамнез или объективный статус."
        )
    if check == "alignment_min":
        return (
            f"Доля совпадения блока «{crit.get('block')}» с протоколом "
            "не ниже порога из методики."
        )
    if check == "no_manual_review":
        return (
            "Проверяется, не направлен ли случай в очередь ручной проверки "
            "из-за неопределённости модели (это не доказанный red flag)."
        )
    return "Автоматическая проверка по правилам постановления № 55."


def evaluate_reg55(case: dict) -> dict:
    """Оценка одного КЗ по критериям № 55.

    Возвращает regulatory_compliance_pct (доля выполненных из применимых),
    списки passed/failed, критические (P0) дефекты, разбивку по группам
    и полный `criteria` (каждый пункт: verdict, point, пояснение).
    """
    prepared = prepare_case_for_reg55(case if isinstance(case, dict) else {})
    reg = _load_reg()
    criteria = reg.get("criteria") or []
    thresholds = reg.get("thresholds") or {}
    missing_file = not criteria and not REG_PATH.is_file()

    passed = 0
    total = 0
    na = 0
    failed: list[dict] = []
    critical_failed: list[dict] = []
    by_group: dict[str, dict[str, int]] = {}
    criteria_detail: list[dict] = []

    for crit in criteria:
        group = str(crit.get("group") or "прочее")
        g = by_group.setdefault(group, {"passed": 0, "total": 0, "na": 0})
        verdict = _eval_criterion(crit, prepared, thresholds)
        how = _how_checked_ru(crit)
        wrong = _whats_wrong_ru(crit, verdict, prepared)
        in_formula = crit.get("score_eligible") is not False
        detail = {
            "id": crit.get("id"),
            "title": crit.get("title"),
            "point": crit.get("point"),
            "point_no": crit.get("point_no") or crit.get("point"),
            "severity": crit.get("severity"),
            "check": crit.get("check"),
            "group": group,
            "verdict": verdict,
            "verdict_ru": {"pass": "выполнен", "fail": "не выполнен", "na": "не применим"}.get(
                verdict, verdict
            ),
            "how_checked_ru": how,
            "whats_wrong_ru": wrong,
            "score": 1.0 if verdict == "pass" else (0.0 if verdict == "fail" else None),
            "in_formula": in_formula and verdict != "na",
            "score_eligible": in_formula,
        }
        criteria_detail.append(detail)
        if not in_formula or verdict == "na":
            na += 1
            g["na"] = int(g.get("na") or 0) + 1
            continue
        total += 1
        g["total"] += 1
        if verdict == "pass":
            passed += 1
            g["passed"] += 1
        else:
            item = {
                "id": crit.get("id"),
                "title": crit.get("title"),
                "point": crit.get("point"),
                "point_no": crit.get("point_no"),
                "severity": crit.get("severity"),
                "check": crit.get("check"),
                "how_checked_ru": how,
                "whats_wrong_ru": wrong,
            }
            failed.append(item)
            if crit.get("severity") == "P0":
                critical_failed.append(item)

    pct = round(100.0 * passed / total, 1) if total else None
    meta = regulation_meta()
    return {
        "regulatory_compliance_pct": pct,
        "passed": passed,
        "total": total,
        "applicable": total,
        "na": na,
        "failed": failed,
        "critical_failed": critical_failed,
        "has_p0_defect": any(f.get("severity") == "P0" for f in failed),
        "by_group": by_group,
        "criteria": criteria_detail,
        "formula_ru": (
            "Средний балл №55 = 100 × (выполненные пункты) / "
            "(применимые пункты; «не применим» и служебные флаги не в знаменателе)"
        ),
        "regulation_id": meta.get("regulation_id"),
        "regulation_title": meta.get("regulation_title"),
        "note_ru": (
            (
                "Файл критериев не найден в образе: data/regulations/mz_2021_55.json. "
                "Процент оси regulatory со склада может быть доступен отдельно."
            )
            if missing_file
            else (
                "Оценка по проверяемым пунктам прил. 2 пост. МЗ № 55 для клинического приёма; "
                "не полная официальная экспертиза организации."
            )
        ),
    }


def format_failed_criteria_ru(items: list[dict], *, limit: int = 6) -> str:
    """Список невыполненных критериев для detail_ru замечания."""
    lines: list[str] = []
    for item in (items or [])[:limit]:
        title = str(item.get("title") or item.get("id") or "критерий")
        point = str(item.get("point") or "").strip()
        sev = str(item.get("severity") or "").strip()
        how = str(item.get("how_checked_ru") or "").strip()
        bit = title
        if sev:
            bit += f" [{sev}]"
        if point:
            bit += f" · {point}"
        if how:
            bit += f" · {how}"
        lines.append(bit)
    return "; ".join(lines)


def regulation_meta() -> dict:
    reg = _load_reg()
    return {
        "regulation_id": reg.get("id") or "mz_2021_55",
        "regulation_title": reg.get("title") or "",
        "regulation_source": reg.get("source") or "",
        "criteria_total": len(reg.get("criteria") or []),
    }


def attach_reg55_to_detail(
    detail: dict[str, Any] | None,
    *,
    clinical: dict[str, Any] | None = None,
    block_scores: dict[str, Any] | None = None,
    live_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Добавить в case-detail полный разбор №55 и pct в record/axes."""
    out = detail if isinstance(detail, dict) else {"ok": False}
    record = dict(out.get("record") or {})
    case: dict[str, Any] = {}
    if isinstance(live_case, dict):
        case.update(live_case)
    case.update(record)
    if isinstance(clinical, dict):
        case["clinical"] = clinical
    if isinstance(block_scores, dict) and block_scores:
        case["block_scores"] = block_scores
    elif isinstance(case.get("block_scores"), dict):
        pass
    else:
        # alignment-критерии останутся na без L1 block_scores - это честно
        case.setdefault("block_scores", {})
    case.setdefault("status", record.get("status") or out.get("deep_status") or "")
    kind = str(record.get("document_kind") or case.get("document_kind") or "").strip()
    if kind == "consultation":
        kind = "clinical_visit"
    if kind and kind != "clinical_visit":
        axes = dict(out.get("axes") or {})
        out["record"] = record
        out["axes"] = axes
        out["reg55"] = {
            "regulatory_compliance_pct": None,
            "passed": 0,
            "total": 0,
            "applicable": 0,
            "na": 0,
            "failed": [],
            "critical_failed": [],
            "has_p0_defect": False,
            "criteria": [],
            "formula_ru": (
                "Средний балл №55 считается только для типа «Клинический приём» "
                "(clinical_visit); na не в знаменателе."
            ),
            "note_ru": (
                f"Тип документа «{kind}» не оценивается по постановлению № 55 "
                "(нужен clinical_visit)."
            ),
        }
        return out
    reg = evaluate_reg55(case)
    axes = dict(out.get("axes") or {})
    pct = reg.get("regulatory_compliance_pct")
    # Fallback: warehouse axis / record, если JSON критериев не в образе
    # или live-пересчёт дал пусто (не затираем уже известный %).
    if not isinstance(pct, (int, float)):
        for candidate in (
            record.get("reg55_pct"),
            axes.get("regulatory"),
        ):
            if isinstance(candidate, (int, float)):
                pct = float(candidate)
                reg["regulatory_compliance_pct"] = pct
                reg.setdefault(
                    "note_ru",
                    "Показан процент оси regulatory со склада; детализация критериев "
                    "доступна при наличии data/regulations/mz_2021_55.json.",
                )
                break
    if isinstance(pct, (int, float)):
        record["reg55_pct"] = float(pct)
        axes["regulatory"] = float(pct)
    out["record"] = record
    out["axes"] = axes
    out["reg55"] = reg
    return out
