"""Ограничения отчёта B2C по product tier (promo vs basic/plus)."""
from __future__ import annotations

from typing import Any

from .patient_clinic_config import TIER_CATALOG, resolve_tier

_PROMO_QUESTION_LIMIT = 3
_PROMO_BLOCK_LIMIT = 4


def _expand_includes(tier_id: str, _depth: int = 0) -> set[str]:
    """Раскрыть вложенные include (plus → basic → traffic_light/questions/...)."""
    if _depth > 6:
        return set()
    tier = TIER_CATALOG.get(tier_id) or {}
    out: set[str] = set()
    for item in tier.get("includes") or []:
        name = str(item)
        if name in TIER_CATALOG:
            out |= _expand_includes(name, _depth + 1)
        else:
            out.add(name)
    return out


def apply_catalog_tier_limits(
    patient_report: dict[str, Any],
    *,
    catalog_tier_id: str | None,
) -> dict[str, Any]:
    """Урезать отчёт для promo/free preview согласно TIER_CATALOG.includes."""
    if not isinstance(patient_report, dict):
        return patient_report
    tier = resolve_tier(catalog_tier_id)
    tier_id = str(tier.get("tier_id") or "basic")
    includes = _expand_includes(tier_id) or {str(x) for x in (tier.get("includes") or [])}
    out = dict(patient_report)

    has_citations = "citations" in includes
    has_questions = "questions" in includes

    if not has_citations:
        out["protocol_citations"] = []
        out.pop("protocol_summary_panel", None)

    if has_questions:
        qs = list(out.get("questions_for_doctor") or [])
        checklist = list(out.get("action_checklist") or [])
        full_count = max(len(qs), len(checklist))
        # На промо-тарифе показываем только часть вопросов, но честно сообщаем остаток.
        if not has_citations and full_count > _PROMO_QUESTION_LIMIT:
            out["questions_for_doctor"] = qs[:_PROMO_QUESTION_LIMIT]
            out["action_checklist"] = checklist[:_PROMO_QUESTION_LIMIT]
            out["questions_structured"] = list(out.get("questions_structured") or [])[:_PROMO_QUESTION_LIMIT]
            out["questions_truncated"] = True
            out["questions_total"] = full_count
            out["questions_hidden_count"] = full_count - _PROMO_QUESTION_LIMIT
    else:
        out["questions_for_doctor"] = []
        out["action_checklist"] = []
        out.pop("structured_questions", None)
        out.pop("questions_structured", None)

    blocks = list(out.get("blocks") or [])
    if blocks and "blocks" in includes and not has_citations:
        concern = [b for b in blocks if isinstance(b, dict) and b.get("status") != "ok"]
        out["blocks"] = (concern or blocks)[:_PROMO_BLOCK_LIMIT]
        out["blocks_truncated"] = len(blocks) > len(out["blocks"])
        out["blocks_hidden_count"] = max(0, len(blocks) - len(out["blocks"]))

    out["catalog_tier_id"] = tier_id
    out["tier_preview"] = tier_id == "promo"
    return out
