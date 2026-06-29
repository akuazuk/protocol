"""Ограничения отчёта B2C по product tier (promo vs basic/plus)."""
from __future__ import annotations

from typing import Any

from .patient_clinic_config import resolve_tier


def apply_catalog_tier_limits(
    patient_report: dict[str, Any],
    *,
    catalog_tier_id: str | None,
) -> dict[str, Any]:
    """Урезать отчёт для promo/free preview согласно TIER_CATALOG.includes."""
    if not isinstance(patient_report, dict):
        return patient_report
    tier = resolve_tier(catalog_tier_id)
    includes = {str(x) for x in (tier.get("includes") or [])}
    tier_id = str(tier.get("tier_id") or "basic")
    out = dict(patient_report)

    if "citations" not in includes:
        out["protocol_citations"] = []
        out.pop("protocol_summary_panel", None)

    if "questions" in includes:
        qs = list(out.get("questions_for_doctor") or [])
        if len(qs) > 3:
            out["questions_for_doctor"] = qs[:3]
            out["questions_truncated"] = True
    else:
        out["questions_for_doctor"] = []
        out.pop("structured_questions", None)

    blocks = list(out.get("blocks") or [])
    if blocks and "blocks" in includes and "citations" not in includes:
        concern = [b for b in blocks if isinstance(b, dict) and b.get("status") != "ok"]
        out["blocks"] = (concern or blocks)[:4]
        out["blocks_truncated"] = len(blocks) > len(out["blocks"])

    out["catalog_tier_id"] = tier_id
    out["tier_preview"] = tier_id == "promo"
    return out
