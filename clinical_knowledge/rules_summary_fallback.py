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


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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
        from .protocol_summary.icd_index import find_summary_refs_by_icd
        from .protocol_summary.loader import load_summary_by_protocol_id
        from .protocol_summary.summary_to_rules import (
            condition_to_protocol_rules,
            protocol_rule_to_legacy_dict,
        )
    except ImportError:
        return clinical_rules

    max_summaries = max(1, min(6, _env_int("CONSULT_RULES_SUMMARY_FALLBACK_MAX_SUMMARIES", 2)))
    max_rules = max(5, min(120, _env_int("CONSULT_RULES_SUMMARY_FALLBACK_MAX_RULES", 48)))

    facts = clinical_rules.get("consult_facts") or {}
    matched = clinical_rules.get("matched_protocols") or []
    extra: list[dict[str, Any]] = []
    condition_ids: list[str] = []
    seen_summary: set[str] = set()
    seen_rule: set[str] = set()

    refs: list[tuple[str, str]] = []
    for icd in codes[:6]:
        for ref in find_summary_refs_by_icd(icd, limit=2):
            if ref not in refs:
                refs.append(ref)
        if len(refs) >= max_summaries:
            break
    refs = refs[:max_summaries]

    for protocol_id, condition_id in refs:
        if protocol_id in seen_summary:
            continue
        summary = load_summary_by_protocol_id(protocol_id)
        if summary is None:
            continue
        seen_summary.add(protocol_id)
        cond = next((c for c in summary.conditions if c.condition_id == condition_id), None)
        if cond is None:
            continue
        if condition_id not in condition_ids:
            condition_ids.append(condition_id)
        for pr in condition_to_protocol_rules(summary, cond):
            leg = protocol_rule_to_legacy_dict(pr)
            rid = str(leg.get("rule_id") or "")
            if rid and rid in seen_rule:
                continue
            if rid:
                seen_rule.add(rid)
            extra.append(leg)
            if len(extra) >= max_rules:
                break
        if len(extra) >= max_rules:
            break

    if not extra:
        return clinical_rules

    new_check = run_rule_checker(
        facts,
        condition_ids=condition_ids or None,
        matched_protocols=matched if isinstance(matched, list) else None,
        extra_rules=extra,
        include_catalog=False,
    )
    new_pct = new_check.get("rules_compliance_pct")
    if new_pct is None or float(new_pct) <= 0:
        return clinical_rules

    out = dict(clinical_rules)
    merged_rc = dict(rc)
    merged_rc.update(new_check)
    merged_rc["summary_fallback_applied"] = True
    merged_rc["summary_rules_count"] = len(extra)
    merged_rc["summary_fallback_summaries"] = len(seen_summary)
    merged_rc["method"] = str(rc.get("method") or "") + "+summary_icd_fallback"
    out["rules_check"] = merged_rc
    return out
