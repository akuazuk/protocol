"""Детерминированный итог «Ориентировочное соответствие» для проверки КЗ.

При включённом структурном разборе итог не зависит от баллов LLM-критериев
(они остаются для комментариев и таблицы). Один и тот же текст КЗ → один процент.

Гибрид: взвешенная оценка структурного разбора (6 блоков по compliance_weights.yaml)
+ проверка по правилам протокола из RAG-пайплайна. При отсутствии структурного
блока — взвешенное среднее LLM-критериев по клиническим темам + правила.
"""
from __future__ import annotations

import os
import re
from typing import Any

SCORER_VERSION = "2026-05-31.1"

# structured : rules — при наличии обоих компонентов
_BLEND_STRUCTURED_RULES = (0.75, 0.25)
# llm_weighted : rules — fallback без структурного overall
_BLEND_LLM_RULES = (0.60, 0.40)

_CRITERION_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"диагноз", re.I), 0.22),
    (re.compile(r"обслед|диагност|лабор|инструмент", re.I), 0.20),
    (re.compile(r"лечен|терап|назнач|рекоменд", re.I), 0.22),
    (re.compile(r"безопас|противопоказ|риск", re.I), 0.12),
    (re.compile(r"протокол|соответств|клиническ", re.I), 0.14),
    (re.compile(r"документ|оформ|структур|полнот", re.I), 0.10),
]
_CRITERION_DEFAULT_WEIGHT = 0.08


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _clamp_pct(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _round_int(v: float) -> int:
    return int(round(_clamp_pct(v)))


def _extract_structured_score(structured_analysis: dict[str, Any] | None) -> float | None:
    if not structured_analysis:
        return None
    comp = structured_analysis.get("compliance") or {}
    if isinstance(comp.get("overall_score"), (int, float)):
        return float(comp["overall_score"])
    bd = comp.get("score_breakdown") or {}
    val = bd.get("overall_score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _extract_rules_pct(
    clinical_rules: dict[str, Any] | None,
    structured_analysis: dict[str, Any] | None,
) -> float | None:
    if isinstance(clinical_rules, dict):
        rc = clinical_rules.get("rules_check") or {}
        if isinstance(rc, dict):
            v = rc.get("rules_compliance_pct")
            if isinstance(v, (int, float)):
                return _clamp_pct(float(v))
    if structured_analysis:
        rc2 = structured_analysis.get("rules_check") or {}
        if isinstance(rc2, dict):
            v = rc2.get("rules_compliance_pct")
            if isinstance(v, (int, float)):
                return _clamp_pct(float(v))
    return None


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    return sum(s * w for s, w in pairs) / total_w


def _llm_criteria_weighted_score(review: dict[str, Any]) -> float | None:
    crits = review.get("criteria")
    if not isinstance(crits, list):
        return None
    pairs: list[tuple[float, float]] = []
    for c in crits:
        if not isinstance(c, dict):
            continue
        try:
            score = float(c.get("score_pct"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= score <= 100.0):
            continue
        name = str(c.get("name_ru") or c.get("name") or "")
        w = _CRITERION_DEFAULT_WEIGHT
        for pat, pw in _CRITERION_PATTERNS:
            if pat.search(name):
                w = pw
                break
        pairs.append((score, w))
    return _weighted_mean(pairs)


def _apply_safety_cap(
    overall: float,
    structured_analysis: dict[str, Any] | None,
) -> tuple[float, bool]:
    """Снижение при необработанных критических safety / manual_review."""
    if not structured_analysis:
        return overall, False
    comp = structured_analysis.get("compliance") or {}
    if comp.get("overall_status") == "manual_review_required":
        return min(overall, 45.0), True
    for s in comp.get("safety_assessments") or []:
        if not isinstance(s, dict):
            continue
        if s.get("severity") == "critical" and s.get("status") != "handled":
            return min(overall, 35.0), True
    for i in comp.get("critical_issues") or []:
        if isinstance(i, dict) and i.get("severity") in ("critical", "high"):
            return min(overall, 50.0), True
    return overall, False


def _blend_two(
    a: float | None,
    wa: float,
    b: float | None,
    wb: float,
) -> tuple[float | None, str]:
    pairs: list[tuple[float, float]] = []
    if a is not None and wa > 0:
        pairs.append((a, wa))
    if b is not None and wb > 0:
        pairs.append((b, wb))
    if not pairs:
        return None, "empty"
    total = sum(w for _, w in pairs)
    mean = _weighted_mean([(s, w / total) for s, w in pairs])
    if len(pairs) == 2:
        return mean, "both"
    if a is not None:
        return a, "structured_only" if wa >= wb else "rules_only"
    return b, "rules_only"


def apply_hybrid_overall_compliance(
    review: dict[str, Any],
    *,
    structured_analysis: dict[str, Any] | None = None,
    clinical_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Перезаписывает review.overall_compliance_pct детерминированным гибридом."""
    if not isinstance(review, dict):
        return review
    if not _env_bool("CONSULT_OVERALL_HYBRID", True):
        return review

    structured = _extract_structured_score(structured_analysis)
    rules = _extract_rules_pct(clinical_rules, structured_analysis)
    llm_w = _llm_criteria_weighted_score(review)

    components: dict[str, float | None] = {
        "structured": structured,
        "rules": rules,
        "llm_criteria_weighted": llm_w,
    }

    raw: float | None
    method_suffix: str

    if structured is not None:
        raw, method_suffix = _blend_two(
            structured,
            _BLEND_STRUCTURED_RULES[0],
            rules,
            _BLEND_STRUCTURED_RULES[1],
        )
        method = f"hybrid_structured_{method_suffix}"
    elif llm_w is not None or rules is not None:
        raw, method_suffix = _blend_two(
            llm_w,
            _BLEND_LLM_RULES[0],
            rules,
            _BLEND_LLM_RULES[1],
        )
        method = f"hybrid_llm_{method_suffix}"
    else:
        review["overall_compliance_scorer_version"] = SCORER_VERSION
        review["overall_compliance_components"] = {
            k: (_round_int(v) if isinstance(v, (int, float)) else None)
            for k, v in components.items()
        }
        return review

    if raw is None:
        return review

    capped_raw, capped = _apply_safety_cap(raw, structured_analysis)
    overall_int = _round_int(capped_raw)

    review["overall_compliance_pct"] = overall_int
    review["overall_compliance_method"] = method + ("_safety_cap" if capped else "")
    review["overall_compliance_components"] = {
        k: (_round_int(v) if isinstance(v, (int, float)) else None)
        for k, v in components.items()
    }
    review["overall_compliance_scorer_version"] = SCORER_VERSION
    return review
