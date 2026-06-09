"""Fallback правил из Protocol Summary Cards, когда каталог не дал срабатываний."""
from __future__ import annotations

import os
from typing import Any

from .rule_checker import run_rule_checker


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _needs_summary_fallback(rules_check: dict[str, Any]) -> bool:
    pct = rules_check.get("rules_compliance_pct")
    if isinstance(pct, (int, float)) and float(pct) > 0:
        return False
    findings = rules_check.get("findings") or []
    scored = [f for f in findings if not f.get("skipped")]
    if scored and any(f.get("passed") for f in scored):
        return False
    return True


def apply_summary_rules_fallback(
    clinical_rules: dict[str, Any] | None,
    icd_codes: list[str] | None,
) -> dict[str, Any] | None:
    """Дополняет rules_check правилами summary по МКБ, если каталог вернул 0%."""
    if not clinical_rules or not _env_bool("CONSULT_RULES_SUMMARY_FALLBACK", True):
        return clinical_rules
    rc = clinical_rules.get("rules_check") or {}
    if not isinstance(rc, dict) or not _needs_summary_fallback(rc):
        return clinical_rules

    codes = [str(c).upper().strip() for c in (icd_codes or []) if c]
    if not codes:
        return clinical_rules

    try:
        from .protocol_summary.loader import find_conditions_by_icd, find_summary_for_condition
        from .protocol_summary.summary_to_rules import (
            protocol_rule_to_legacy_dict,
            summary_to_protocol_rules,
        )
    except ImportError:
        return clinical_rules

    facts = clinical_rules.get("consult_facts") or {}
    matched = clinical_rules.get("matched_protocols") or []
    extra: list[dict[str, Any]] = []
    seen_summary: set[str] = set()
    seen_rule: set[str] = set()

    for icd in codes[:8]:
        for cond in find_conditions_by_icd(icd)[:4]:
            summary = find_summary_for_condition(cond, usable_only=False)
            if summary is None or summary.protocol_id in seen_summary:
                continue
            seen_summary.add(summary.protocol_id)
            for pr in summary_to_protocol_rules(summary):
                if cond.condition_id and pr.condition_id and pr.condition_id != cond.condition_id:
                    continue
                leg = protocol_rule_to_legacy_dict(pr)
                rid = str(leg.get("rule_id") or "")
                if rid and rid in seen_rule:
                    continue
                if rid:
                    seen_rule.add(rid)
                extra.append(leg)

    if not extra:
        return clinical_rules

    new_check = run_rule_checker(
        facts,
        matched_protocols=matched if isinstance(matched, list) else None,
        extra_rules=extra,
    )
    new_pct = new_check.get("rules_compliance_pct")
    if new_pct is None or float(new_pct) <= 0:
        return clinical_rules

    out = dict(clinical_rules)
    merged_rc = dict(rc)
    merged_rc.update(new_check)
    merged_rc["summary_fallback_applied"] = True
    merged_rc["summary_rules_count"] = len(extra)
    merged_rc["method"] = str(rc.get("method") or "") + "+summary_icd_fallback"
    out["rules_check"] = merged_rc
    return out
