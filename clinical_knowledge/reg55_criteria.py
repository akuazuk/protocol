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


def _fields_present(case: dict) -> dict:
    fp = case.get("fields_present")
    return fp if isinstance(fp, dict) else {}


def _block_score(case: dict, block: str) -> float | None:
    bs = case.get("block_scores")
    if not isinstance(bs, dict):
        return None
    val = bs.get(block)
    return float(val) if isinstance(val, (int, float)) else None


def _icd10_present(case: dict) -> bool:
    # Весь МО/КЗ, не только графа «Диагноз» (план mo-icd-full-document-search).
    from .mo_icd_resolve import resolve_icd_codes_from_mo

    return bool(resolve_icd_codes_from_mo(case).get("present"))


def _diagnosis_substantiated(case: dict) -> bool:
    fp = _fields_present(case)
    return bool(
        fp.get("diagnosis")
        and fp.get("complaints")
        and (fp.get("anamnesis") or fp.get("objective_status"))
    )


def _eval_criterion(crit: dict, case: dict, thresholds: dict) -> str:
    """Возвращает 'pass' | 'fail' | 'na'."""
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
            return "na"  # блок протокола не применялся - не дефект
        thr = float(thresholds.get(crit.get("threshold_key"), 50))
        return "pass" if score >= thr else "fail"
    if check == "no_manual_review":
        return "fail" if str(case.get("status")) == "manual_review_required" else "pass"
    return "na"


def evaluate_reg55(case: dict) -> dict:
    """Оценка одного КЗ по критериям № 55.

    Возвращает regulatory_compliance_pct (доля выполненных из применимых),
    списки passed/failed, критические (P0) дефекты и разбивку по группам.
    """
    reg = _load_reg()
    criteria = reg.get("criteria") or []
    thresholds = reg.get("thresholds") or {}

    passed = 0
    total = 0
    na = 0
    failed: list[dict] = []
    critical_failed: list[dict] = []
    by_group: dict[str, dict[str, int]] = {}

    for crit in criteria:
        group = str(crit.get("group") or "прочее")
        g = by_group.setdefault(group, {"passed": 0, "total": 0})
        verdict = _eval_criterion(crit, case, thresholds)
        if verdict == "na":
            na += 1
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
                "severity": crit.get("severity"),
                "check": crit.get("check"),
                "how_checked_ru": _how_checked_ru(crit),
            }
            failed.append(item)
            if crit.get("severity") == "P0":
                critical_failed.append(item)

    pct = round(100.0 * passed / total, 1) if total else None
    return {
        "regulatory_compliance_pct": pct,
        "passed": passed,
        "total": total,
        "na": na,
        "failed": failed,
        "critical_failed": critical_failed,
        "has_p0_defect": any(f.get("severity") == "P0" for f in failed),
        "by_group": by_group,
    }


def _how_checked_ru(crit: dict) -> str:
    check = crit.get("check")
    if check == "field_present":
        field = crit.get("field") or "поле"
        return f"Проверяется наличие заполненного поля «{field}» в МО/КЗ."
    if check == "icd10_present":
        return (
            "Код МКБ-10 (буква + 2 цифры) ищется по всему тексту МО/КЗ, "
            "не только в графе «Диагноз»."
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
