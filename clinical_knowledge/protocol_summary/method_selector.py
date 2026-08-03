"""Выбор режима legacy / summary / hybrid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from . import config as _cfg_mod
from .config import AnalysisMode
from .loader import load_summary_by_protocol_id
from .schema import ProtocolSummary
from .validator import summary_is_usable

PrimarySource = Literal["legacy", "summary", "none"]

_REVIEW_RANK = {"not_reviewed": 0, "needs_review": 1, "reviewed": 2, "approved": 3}


@dataclass
class AnalysisPlan:
    mode: AnalysisMode
    enabled: bool
    primary_source: PrimarySource
    use_legacy: bool
    use_summary: bool
    fallback_to_legacy: bool
    compare_with_legacy: bool
    usable_summaries: list[ProtocolSummary] = field(default_factory=list)
    summary_rules_count: int = 0
    notes: list[str] = field(default_factory=list)
    summary_diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _match_summaries(protocol_ids: list[str]) -> list[ProtocolSummary]:
    out: list[ProtocolSummary] = []
    seen: set[str] = set()
    for pid in protocol_ids:
        if not pid or pid in seen:
            continue
        s = load_summary_by_protocol_id(pid)
        if s and summary_is_usable(s):
            out.append(s)
            seen.add(pid)
    return out


def _dedupe_summaries(*lists: list[ProtocolSummary]) -> list[ProtocolSummary]:
    seen: set[str] = set()
    out: list[ProtocolSummary] = []
    for lst in lists:
        for s in lst:
            if s.protocol_id not in seen:
                out.append(s)
                seen.add(s.protocol_id)
    return out


def resolve_analysis_plan(
    *,
    mode: AnalysisMode | None = None,
    matched_protocol_ids: list[str] | None = None,
    discovered_summaries: list[ProtocolSummary] | None = None,
    summary_diagnostics: list[dict[str, Any]] | None = None,
    enabled: bool | None = None,
) -> AnalysisPlan:
    cfg = _cfg_mod.protocol_summary_config
    mode = mode or cfg.mode
    enabled = cfg.enabled if enabled is None else enabled
    matched_protocol_ids = matched_protocol_ids or []
    discovered_summaries = discovered_summaries or []
    summary_diagnostics = summary_diagnostics or []

    if not enabled or mode == "legacy":
        return AnalysisPlan(
            mode="legacy",
            enabled=False,
            primary_source="legacy",
            use_legacy=True,
            use_summary=False,
            fallback_to_legacy=True,
            compare_with_legacy=False,
            notes=["protocol summary disabled or legacy mode"],
        )

    usable = _dedupe_summaries(
        _match_summaries(matched_protocol_ids),
        discovered_summaries,
    )
    summary_ok = len(usable) > 0
    notes: list[str] = list(summary_diagnostics and [] or [])

    if not summary_ok:
        not_found = [d for d in summary_diagnostics if d.get("match_reasons") == ["not_found"]]
        if not_found:
            notes.append(not_found[0].get("detail") or "summary not found by ICD/diagnosis/path")

    if mode == "summary":
        if summary_ok:
            return AnalysisPlan(
                mode="summary",
                enabled=True,
                primary_source="summary",
                use_legacy=False,
                use_summary=True,
                fallback_to_legacy=cfg.fallback_to_legacy,
                compare_with_legacy=False,
                usable_summaries=usable,
                summary_rules_count=sum(len(s.conditions) for s in usable),
                notes=notes,
                summary_diagnostics=summary_diagnostics,
            )
        if cfg.fallback_to_legacy:
            notes.append("summary missing/invalid → fallback to legacy")
            return AnalysisPlan(
                mode="summary",
                enabled=True,
                primary_source="legacy",
                use_legacy=True,
                use_summary=False,
                fallback_to_legacy=True,
                compare_with_legacy=False,
                notes=notes,
                summary_diagnostics=summary_diagnostics,
            )
        notes.append("insufficient_protocol_data: no valid summary")
        return AnalysisPlan(
            mode="summary",
            enabled=True,
            primary_source="none",
            use_legacy=False,
            use_summary=False,
            fallback_to_legacy=False,
            compare_with_legacy=False,
            notes=notes,
            summary_diagnostics=summary_diagnostics,
        )

    # hybrid
    if summary_ok:
        notes.append("hybrid: summary primary, legacy for fallback/evidence")
        return AnalysisPlan(
            mode="hybrid",
            enabled=True,
            primary_source="summary",
            use_legacy=True,
            use_summary=True,
            fallback_to_legacy=True,
            compare_with_legacy=cfg.compare_with_legacy,
            usable_summaries=usable,
            summary_rules_count=sum(len(s.conditions) for s in usable),
            notes=notes,
            summary_diagnostics=summary_diagnostics,
        )
    notes.append("hybrid: no valid summary → legacy only")
    return AnalysisPlan(
        mode="hybrid",
        enabled=True,
        primary_source="legacy",
        use_legacy=True,
        use_summary=False,
        fallback_to_legacy=True,
        compare_with_legacy=False,
        notes=notes,
        summary_diagnostics=summary_diagnostics,
    )


def _rule_key(rule: dict[str, Any]) -> str:
    rt = rule.get("rule_type") or ""
    rid = rule.get("rule_id") or rule.get("exam") or rule.get("keyword") or ""
    cid = rule.get("condition_id") or ""
    return f"{cid}::{rt}::{rid}"


def _summary_priority(rule: dict[str, Any], summaries: dict[str, ProtocolSummary]) -> int:
    if rule.get("rule_source") != "summary":
        return 0
    score = 20
    sid = str(rule.get("summary_id") or rule.get("source", {}).get("protocol_id") or "")
    summary = summaries.get(sid)
    if summary:
        score += _REVIEW_RANK.get(summary.review_status, 0) * 10
    src = rule.get("source") or {}
    if src.get("table_index") is not None or src.get("section_type") == "table":
        score += 25
    if rule.get("generated_from_summary"):
        score += 5
    return score


def _legacy_priority(rule: dict[str, Any]) -> int:
    if rule.get("rule_source") == "table":
        return 15
    rt = str(rule.get("rule_type") or "")
    if rt == "keyword_presence":
        return 5
    return 10


def merge_rules_for_plan(
    plan: AnalysisPlan,
    legacy_rules: list[dict[str, Any]],
    summary_rules_dicts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Объединяет legacy и summary rules; summary с большим приоритетом побеждает."""
    summary_rules_dicts = list(summary_rules_dicts or [])
    summaries_map = {s.protocol_id: s for s in plan.usable_summaries}
    meta: dict[str, Any] = {
        "analysis_mode": plan.mode,
        "protocol_summary_used": plan.use_summary and bool(summary_rules_dicts),
        "fallback_to_legacy": plan.primary_source == "legacy" and plan.mode != "legacy",
        "summary_protocol_ids": [s.protocol_id for s in plan.usable_summaries],
        "rule_conflicts": [],
        "suppressed_legacy_rule_ids": [],
    }

    if not plan.use_summary or not summary_rules_dicts:
        if plan.mode == "summary" and not summary_rules_dicts:
            meta["protocol_summary_used"] = False
        return legacy_rules if plan.use_legacy else [], meta

    combined: list[dict[str, Any]] = []
    summary_by_key: dict[str, dict[str, Any]] = {}
    for sr in summary_rules_dicts:
        sr = dict(sr)
        sr.setdefault("rule_source", "summary")
        k = _rule_key(sr)
        summary_by_key[k] = sr
        combined.append(sr)

    suppressed: set[str] = set()
    conflicts: list[dict[str, Any]] = []

    if plan.mode == "hybrid" and plan.use_legacy:
        seen = set(summary_by_key.keys())
        for lr in legacy_rules:
            lr = dict(lr)
            lr.setdefault("rule_source", "legacy")
            k = _rule_key(lr)
            if k in seen:
                sr = summary_by_key[k]
                sp = _summary_priority(sr, summaries_map)
                lp = _legacy_priority(lr)
                if sp >= lp:
                    suppressed.add(str(lr.get("rule_id") or ""))
                    conflicts.append({
                        "rule_key": k,
                        "resolution": "summary_wins",
                        "summary_rule_id": sr.get("rule_id"),
                        "legacy_rule_id": lr.get("rule_id"),
                        "manual_review": sp > 0 and lp > 0 and abs(sp - lp) < 15,
                    })
                    continue
                conflicts.append({
                    "rule_key": k,
                    "resolution": "legacy_wins",
                    "summary_rule_id": sr.get("rule_id"),
                    "legacy_rule_id": lr.get("rule_id"),
                    "manual_review": True,
                })
            combined.append(lr)
            seen.add(k)
    elif plan.mode == "summary":
        combined = list(summary_rules_dicts)

    meta["rule_conflicts"] = conflicts
    meta["suppressed_legacy_rule_ids"] = sorted(x for x in suppressed if x)
    meta["rules_count_by_source"] = {
        "summary": sum(1 for r in combined if r.get("rule_source") == "summary"),
        "legacy": sum(1 for r in combined if r.get("rule_source") != "summary"),
    }
    return combined, meta
