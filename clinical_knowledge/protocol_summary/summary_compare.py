"""Сравнение legacy, summary и hybrid на одном КЗ."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..consult_analysis import analyze_consultation_text

ROOT = Path(__file__).resolve().parents[2]


def compliance_metrics(compliance: dict[str, Any]) -> dict[str, Any]:
    """Извлекает метрики из JSON compliance для method comparison."""
    safety = compliance.get("safety_assessments") or []
    red_flag_types = (
        "possible_malignancy", "thrombosis", "severe_infection",
        "systemic_autoimmune", "drug_safety", "urgent_referral",
    )
    red_flags = sum(1 for s in safety if s.get("issue_type") in red_flag_types)
    evidence = compliance.get("evidence_map") or []
    summary_ev = sum(1 for e in evidence if e.get("rule_source") == "summary")
    legacy_ev = sum(1 for e in evidence if e.get("rule_source") == "legacy")
    return {
        "overall_score": compliance.get("overall_score"),
        "confidence_score": compliance.get("confidence_score"),
        "status": compliance.get("overall_status"),
        "matched_protocols": len(compliance.get("matched_protocols") or []),
        "matched_conditions": len(compliance.get("summary_diagnostics", [{}])[0].get("condition_ids", []))
        if compliance.get("summary_diagnostics")
        else len({
            e.get("rule_id", "").split("__")[1]
            for e in evidence
            if e.get("rule_source") == "summary" and "__" in str(e.get("rule_id", ""))
        }),
        "critical_issues": len(compliance.get("critical_issues") or []),
        "major_issues": len(compliance.get("major_issues") or []),
        "missing_required_exams": sum(
            1 for e in (compliance.get("exam_assessments") or [])
            if e.get("status") == "missing_required"
        ),
        "treatment_issues": sum(
            1 for t in (compliance.get("treatment_assessments") or [])
            if t.get("status") not in ("matches_protocol", "not_assessed", None)
        ),
        "red_flags": red_flags,
        "manual_review_required": compliance.get("overall_status") == "manual_review_required",
        "source_refs_count": len(compliance.get("source_refs") or [])
        + len(compliance.get("summary_source_refs") or [])
        + len(compliance.get("legacy_source_refs") or []),
        "explainability_score": round(
            100.0 * (summary_ev + legacy_ev) / max(len(evidence), 1), 1,
        ) if evidence else None,
        "protocol_summary_used": compliance.get("protocol_summary_used"),
        "fallback_to_legacy": compliance.get("fallback_to_legacy"),
        "rules_count_by_source": compliance.get("rules_count_by_source"),
        "analysis_mode": compliance.get("analysis_mode"),
    }


def _score(compliance: dict[str, Any]) -> float | None:
    return compliance.get("overall_score")


def compare_modes_on_text(
    raw_text: str,
    *,
    consultation_id: str = "compare",
) -> dict[str, Any]:
    """Прогон legacy (default) и явных режимов если включены env."""
    import os

    prev = dict(os.environ)
    results: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    try:
        os.environ["PROTOCOL_SUMMARY_ENABLED"] = "0"
        legacy = analyze_consultation_text(raw_text, consultation_id=consultation_id, with_markdown=False)
        results["legacy"] = legacy.get("compliance") or {}
        metrics["legacy"] = compliance_metrics(results["legacy"])

        for mode in ("summary", "hybrid"):
            os.environ["PROTOCOL_SUMMARY_ENABLED"] = "1"
            os.environ["PROTOCOL_SUMMARY_MODE"] = mode
            from .config import ProtocolSummaryConfig
            from . import loader as _loader

            _loader.clear_protocol_summary_cache()
            import clinical_knowledge.protocol_summary.config as cfg_mod

            cfg_mod.protocol_summary_config = ProtocolSummaryConfig.from_env()
            res = analyze_consultation_text(
                raw_text,
                consultation_id=f"{consultation_id}_{mode}",
                with_markdown=False,
                analysis_mode=mode,
            )
            results[mode] = res.get("compliance") or {}
            metrics[mode] = compliance_metrics(results[mode])
    finally:
        os.environ.clear()
        os.environ.update(prev)

    l_score = _score(results.get("legacy") or {})
    s_score = _score(results.get("summary") or {})
    h_score = _score(results.get("hybrid") or {})
    same = l_score == s_score if l_score is not None and s_score is not None else None
    return {
        "legacy_score": l_score,
        "summary_score": s_score,
        "hybrid_score": h_score,
        "score_delta_summary": (s_score - l_score) if s_score is not None and l_score is not None else None,
        "same_decision_legacy_summary": same,
        "metrics": metrics,
        "results": results,
    }


def write_comparison_report(
    comparison: dict[str, Any],
    out_path: Path,
    *,
    consultation_id: str = "compare",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Method comparison: {consultation_id}",
        "",
        f"- legacy score: {comparison.get('legacy_score')}",
        f"- summary score: {comparison.get('summary_score')}",
        f"- hybrid score: {comparison.get('hybrid_score')}",
        f"- delta (summary-legacy): {comparison.get('score_delta_summary')}",
        f"- same decision: {comparison.get('same_decision_legacy_summary')}",
        "",
    ]
    for mode, m in (comparison.get("metrics") or {}).items():
        lines.append(f"## {mode}")
        for k, v in m.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def append_batch_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    fieldnames = [
        "consultation_id", "legacy_score", "summary_score", "hybrid_score",
        "score_delta", "same_decision",
        "summary_used", "fallback_used", "critical_issues", "red_flags",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
