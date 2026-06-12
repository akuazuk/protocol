"""Контекст кабинета методиста: compliance КЗ (без подписи/ЦИСЗ) для UI и ML."""
from __future__ import annotations

from typing import Any

from .privacy import redact_kz_text_for_display

STRUCTURED_BLOCK_ROWS: list[dict[str, Any]] = [
    {
        "key": "documentation_score",
        "fallbacks": ["structural_score", "documentation_quality_score"],
        "label_ru": "Оформление КЗ",
    },
    {"key": "patient_data_score", "fallbacks": [], "label_ru": "Данные пациента"},
    {
        "key": "protocol_applicability_score",
        "fallbacks": ["protocol_match_score"],
        "label_ru": "Применимость протокола",
    },
    {"key": "diagnosis_score", "fallbacks": [], "label_ru": "Диагноз"},
    {"key": "required_exams_score", "fallbacks": [], "label_ru": "Обследования"},
    {"key": "treatment_score", "fallbacks": [], "label_ru": "Лечение"},
    {"key": "safety_score", "fallbacks": [], "label_ru": "Безопасность"},
    {"key": "follow_up_score", "fallbacks": [], "label_ru": "Контроль"},
]


def _score_from_breakdown(bd: dict[str, Any], key: str, fallbacks: list[str]) -> float | None:
    for k in [key, *fallbacks]:
        v = bd.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def extract_structured_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") or {}
    bd = comp.get("score_breakdown") or {}
    blocks: list[dict[str, Any]] = []
    for row in STRUCTURED_BLOCK_ROWS:
        score = _score_from_breakdown(bd, row["key"], row.get("fallbacks") or [])
        blocks.append(
            {
                "key": row["key"],
                "label_ru": row["label_ru"],
                "score_pct": score,
            }
        )
    return blocks


def _rules_compliance_pct(result: dict[str, Any]) -> float | None:
    cr = result.get("clinical_rules") or {}
    rc = cr.get("rules_check") or {}
    pct = rc.get("rules_compliance_pct")
    if pct is not None:
        return float(pct)
    comp = (result.get("structured_analysis") or {}).get("compliance") or {}
    bd = comp.get("score_breakdown") or {}
    if isinstance(bd, dict) and bd.get("protocol_rules") is not None:
        return float(bd["protocol_rules"])
    return None


def _llm_criteria_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    rev = result.get("review") or {}
    out: list[dict[str, Any]] = []
    for c in rev.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        score = c.get("score_pct")
        out.append(
            {
                "name_ru": (c.get("name_ru") or c.get("name") or "")[:120],
                "score_pct": float(score) if score is not None else None,
                "criterion_id": (c.get("criterion_id") or c.get("id") or "")[:80],
            }
        )
    return out


def _rules_findings(result: dict[str, Any]) -> list[dict[str, Any]]:
    cr = result.get("clinical_rules") or {}
    rc = cr.get("rules_check") or {}
    findings: list[dict[str, Any]] = []
    for f in rc.get("findings") or []:
        if not isinstance(f, dict) or f.get("skipped"):
            continue
        findings.append(
            {
                "rule_id": f.get("rule_id") or "",
                "title_ru": f.get("title_ru") or f.get("message_ru") or "",
                "passed": f.get("passed"),
                "message_ru": (f.get("message_ru") or "")[:240],
            }
        )
    return findings


def build_methodist_review_context(result: dict[str, Any], full_text: str) -> dict[str, Any]:
    """Сводка для UI методиста: КЗ без ФИО + те же оси оценки, что в обычном режиме."""
    rev = result.get("review") or {}
    sa = result.get("structured_analysis") or {}
    comp = sa.get("compliance") or {}
    comp_parts = rev.get("overall_compliance_components") or {}
    cr = result.get("clinical_rules") or {}

    matched: list[str] = []
    for mp in cr.get("matched_protocols") or []:
        if isinstance(mp, dict) and mp.get("path"):
            matched.append(str(mp["path"]))
    if not matched:
        for p in result.get("retrieval_paths") or []:
            matched.append(str(p))

    blocks = extract_structured_blocks(result)
    structured_pct = comp_parts.get("structured")
    if structured_pct is None:
        structured_pct = comp.get("overall_score")
    rules_pct = comp_parts.get("rules")
    if rules_pct is None:
        rules_pct = _rules_compliance_pct(result)
    overall_pct = rev.get("overall_compliance_pct")
    if overall_pct is None:
        overall_pct = comp.get("overall_score") or result.get("overall_score")

    return {
        "focus": "protocol_compliance",
        "exclude_from_review": ["send_gate", "sign_decision", "cisz_readiness"],
        "kz_text_display": redact_kz_text_for_display(full_text),
        "compliance": {
            "overall_pct": overall_pct,
            "structured_pct": structured_pct,
            "rules_pct": rules_pct,
            "overall_status": comp.get("overall_status") or result.get("overall_status") or "",
            "blocks": blocks,
        },
        "llm_criteria": _llm_criteria_summary(result),
        "rules_findings": _rules_findings(result),
        "critical_issues": comp.get("critical_issues") or [],
        "matched_protocol_paths": matched[:12],
        "retrieval_top_paths": list(result.get("retrieval_paths") or [])[:5],
        "training_record_hint": {
            "autolog_event": "kz_analysis",
            "review_event": "analysis_review",
            "gold_fields": [
                "kz_compliance_gold",
                "verdict",
                "rating",
                "tags",
                "block_overrides",
                "overrides",
                "retrieval_fix",
            ],
        },
    }


def structured_block_scores_dict(result: dict[str, Any]) -> dict[str, float | None]:
    return {b["key"]: b.get("score_pct") for b in extract_structured_blocks(result)}
