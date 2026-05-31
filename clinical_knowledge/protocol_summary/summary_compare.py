"""Сравнение legacy, summary и hybrid на одном КЗ."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..consult_analysis import analyze_consultation_text

ROOT = Path(__file__).resolve().parents[2]


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
    try:
        os.environ["PROTOCOL_SUMMARY_ENABLED"] = "0"
        legacy = analyze_consultation_text(raw_text, consultation_id=consultation_id, with_markdown=False)
        results["legacy"] = legacy.get("compliance") or {}

        for mode in ("summary", "hybrid"):
            os.environ["PROTOCOL_SUMMARY_ENABLED"] = "1"
            os.environ["PROTOCOL_SUMMARY_MODE"] = mode
            from .config import ProtocolSummaryConfig
            from . import loader as _loader

            _loader.clear_protocol_summary_cache()
            # refresh config
            import clinical_knowledge.protocol_summary.config as cfg_mod

            cfg_mod.protocol_summary_config = ProtocolSummaryConfig.from_env()
            res = analyze_consultation_text(
                raw_text,
                consultation_id=f"{consultation_id}_{mode}",
                with_markdown=False,
                analysis_mode=mode,
            )
            results[mode] = res.get("compliance") or {}
    finally:
        os.environ.clear()
        os.environ.update(prev)

    l_score = _score(results.get("legacy") or {})
    s_score = _score(results.get("summary") or {})
    h_score = _score(results.get("hybrid") or {})
    same = l_score == h_score if l_score is not None and s_score is not None else None
    return {
        "legacy_score": l_score,
        "summary_score": s_score,
        "hybrid_score": h_score,
        "score_delta_summary": (s_score - l_score) if s_score is not None and l_score is not None else None,
        "same_decision_legacy_summary": same,
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
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def append_batch_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "consultation_id", "legacy_score", "summary_score", "hybrid_score",
                "score_delta", "same_decision",
            ],
        )
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
