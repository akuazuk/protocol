"""Выбор режима legacy / summary / hybrid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from . import config as _cfg_mod
from .config import AnalysisMode, protocol_summary_config
from .loader import load_summary_by_protocol_id
from .schema import ProtocolSummary
from .validator import summary_is_usable

PrimarySource = Literal["legacy", "summary", "none"]


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


def _match_summaries(protocol_ids: list[str]) -> list[ProtocolSummary]:
    out: list[ProtocolSummary] = []
    for pid in protocol_ids:
        s = load_summary_by_protocol_id(pid)
        if s and summary_is_usable(s):
            out.append(s)
    return out


def resolve_analysis_plan(
    *,
    mode: AnalysisMode | None = None,
    matched_protocol_ids: list[str] | None = None,
    enabled: bool | None = None,
) -> AnalysisPlan:
    cfg = _cfg_mod.protocol_summary_config
    mode = mode or cfg.mode
    enabled = cfg.enabled if enabled is None else enabled
    matched_protocol_ids = matched_protocol_ids or []

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

    usable = _match_summaries(matched_protocol_ids)
    summary_ok = len(usable) > 0
    notes: list[str] = []

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
            )
        return AnalysisPlan(
            mode="summary",
            enabled=True,
            primary_source="none",
            use_legacy=False,
            use_summary=False,
            fallback_to_legacy=False,
            compare_with_legacy=False,
            notes=["insufficient_protocol_data: no valid summary"],
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
    )


def merge_rules_for_plan(
    plan: AnalysisPlan,
    legacy_rules: list[dict[str, Any]],
    summary_rules_dicts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Объединяет legacy и summary rules согласно plan."""
    meta: dict[str, Any] = {
        "analysis_mode": plan.mode,
        "protocol_summary_used": plan.use_summary,
        "fallback_to_legacy": plan.primary_source == "legacy" and plan.mode != "legacy",
        "summary_protocol_ids": [s.protocol_id for s in plan.usable_summaries],
    }
    if not plan.use_summary:
        return legacy_rules, meta
    combined = list(summary_rules_dicts or [])
    if plan.mode == "hybrid":
        seen = {_rule_key(r) for r in combined}
        for lr in legacy_rules:
            k = _rule_key(lr)
            if k not in seen:
                lr = dict(lr)
                lr.setdefault("rule_source", "legacy")
                combined.append(lr)
                seen.add(k)
    elif plan.mode == "summary":
        combined = list(summary_rules_dicts or [])
    return combined, meta


def _rule_key(rule: dict[str, Any]) -> str:
    return f"{rule.get('rule_type')}::{rule.get('rule_id') or rule.get('exam') or rule.get('keyword')}"
