"""Оценка КЗ по критериям Постановления МЗ РБ 21.05.2021 № 55 (case-level).

Детерминированно (без LLM). Считает долю выполненных критериев уровня случая
и список невыполненных со ссылкой на пункт постановления. Работает по полям
`case` из cases.jsonl (`fields_present`, `block_scores`, `diagnosis_short`,
`status`), поэтому доступно и при rebuild summary без доступа к БД.

stdlib-only: модуль импортируется батч-скриптом без Pydantic.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REG_PATH = ROOT / "data" / "regulations" / "mz_2021_55.json"

# МКБ-10: латинская буква + 2 цифры (+ опционально .цифра[цифра])
_ICD10_RE = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?\b")


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
    txt = str(case.get("diagnosis_short") or "")
    return bool(_ICD10_RE.search(txt))


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
            }
            failed.append(item)
            if crit.get("severity") in ("P0", "P1"):
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


def regulation_meta() -> dict:
    reg = _load_reg()
    return {
        "regulation_id": reg.get("id") or "mz_2021_55",
        "regulation_title": reg.get("title") or "",
        "regulation_source": reg.get("source") or "",
        "criteria_total": len(reg.get("criteria") or []),
    }
