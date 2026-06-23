"""Фильтрация правил по подобранным протоколам (снижение ложных срабатываний)."""
from __future__ import annotations

from typing import Any


def matched_source_paths(matched_protocols: list[dict[str, Any]] | None) -> set[str]:
    out: set[str] = set()
    for mp in matched_protocols or []:
        if not isinstance(mp, dict):
            continue
        sp = (mp.get("source_path") or "").replace("\\", "/").strip()
        if sp:
            out.add(sp)
    return out


def filter_rules_for_matched_protocols(
    rules: list[dict[str, Any]],
    matched_protocols: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Оставить ручные правила; авто - только с source_path из top matched PDF."""
    paths = matched_source_paths(matched_protocols)
    if not paths:
        return _dedupe_rule_types(rules)

    filtered: list[dict[str, Any]] = []
    for rule in rules:
        if rule.get("rule_source") == "summary" or rule.get("generated_from_summary"):
            filtered.append(rule)
            continue
        if not rule.get("auto_extracted"):
            filtered.append(rule)
            continue
        src = rule.get("source") or {}
        rule_path = (src.get("source_path") or "").replace("\\", "/").strip()
        if rule_path and rule_path in paths:
            filtered.append(rule)

    if not filtered:
        # fallback: только ручные правила, если авто не совпали с matched PDF
        filtered = [r for r in rules if not r.get("auto_extracted")]

    return _dedupe_rule_types(filtered)


def _dedupe_rule_types(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Один diagnosis_formula и один diagnostic_criterion на нозологию (первый после фильтра)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rule in rules:
        rt = str(rule.get("rule_type") or "")
        if rt in ("diagnosis_formula", "diagnostic_criterion"):
            if rt in seen:
                continue
            seen.add(rt)
        out.append(rule)
    return out
