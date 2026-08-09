#!/usr/bin/env python3
"""Choose an exploratory provisional methodology from proxy or C7 aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _endpoint_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    ranking = list(
        payload.get("ranking_by_proxy_pr_auc")
        or payload.get("ranking_by_gold_pr_auc")
        or []
    )
    metrics = payload.get("metrics") or {}
    labeled_n = int(
        payload.get("proxy_labeled_n")
        if payload.get("proxy_labeled_n") is not None
        else payload.get("gold_labeled_n")
        or 0
    )
    bad_n = int(
        payload.get("proxy_bad_n")
        if payload.get("proxy_bad_n") is not None
        else payload.get("gold_bad_n")
        or 0
    )
    return {
        "ranking": ranking,
        "metrics": metrics,
        "labeled_n": labeled_n,
        "bad_n": bad_n,
    }


def choose_provisional(
    report: Mapping[str, Any],
    *,
    analysis: str | None = None,
    required_next_gate: str | None = None,
) -> dict[str, Any]:
    gold_kind = str(report.get("gold_kind") or "")
    if analysis is None:
        analysis = (
            "c8b_c7_llm_proxy_gold_provisional"
            if gold_kind.startswith("llm_proxy")
            else "agent_proxy_provisional_c8a"
        )
    if required_next_gate is None:
        required_next_gate = (
            "confirmatory_cohort_and_owner_review"
            if gold_kind.startswith("llm_proxy")
            else "independent_methodist_labels"
        )
    endpoints: dict[str, Any] = {}
    for endpoint in ("dx", "plan"):
        view = _endpoint_view(report.get("endpoints", {}).get(endpoint) or {})
        ranking = view["ranking"]
        metrics = view["metrics"]
        top_name = ranking[0] if ranking else None
        top = metrics.get(top_name) if top_name else None
        bad_n = view["bad_n"]
        labeled_n = view["labeled_n"]
        stable = bool(
            top
            and bad_n >= 8
            and labeled_n >= 20
            and (top.get("pr_auc_bad") or 0) >= 0.5
            and (top.get("roc_auc_bad") or 0) >= 0.55
        )
        endpoints[endpoint] = {
            "top_candidate": top_name,
            "top_metrics": (
                {
                    key: top.get(key)
                    for key in (
                        "n",
                        "proxy_bad_n",
                        "mae",
                        "spearman",
                        "roc_auc_bad",
                        "pr_auc_bad",
                        "pr_auc_bad_ci95",
                        "classification_at_55",
                    )
                }
                if isinstance(top, dict)
                else None
            ),
            "ranking_top3": ranking[:3],
            "proxy_labeled_n": labeled_n,
            "proxy_bad_n": bad_n,
            "stable_enough_for_provisional": stable,
            "decision": (
                f"provisional_shadow:{top_name}"
                if stable and top_name
                else "no_stable_provisional"
            ),
        }
    return {
        "schema_version": 1,
        "analysis": analysis,
        "gold_kind": gold_kind or None,
        "proxy_models": list(report.get("proxy_models") or []),
        "endpoints": endpoints,
        "production_rollout": {
            "allowed": False,
            "reason": "proxy_not_human_gold",
            "required_next_gate": required_next_gate,
        },
        "shadow_recommendation": {
            "dx": endpoints["dx"]["decision"],
            "plan": endpoints["plan"]["decision"],
            "note": (
                "Shadow only. Do not change action queue, thresholds, SSOT, or "
                "warehouse until confirmatory gate and owner review. "
                "LLM-proxy gold is not human methodist gold."
            ),
        },
        "phi_check": {
            "contains_source_identifiers": False,
            "contains_sample_identifiers": False,
            "contains_clinical_text": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-eval", type=Path, help="C7A/C9A proxy aggregate JSON")
    parser.add_argument("--c7-eval", type=Path, help="C7 against C6/C6B labels JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", default="")
    args = parser.parse_args()
    if bool(args.proxy_eval) == bool(args.c7_eval):
        raise SystemExit("provide exactly one of --proxy-eval or --c7-eval")
    source = args.proxy_eval or args.c7_eval
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("eval report must be a JSON object")
    chosen = choose_provisional(
        report,
        analysis=args.analysis or None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(chosen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis": chosen["analysis"],
                "dx": chosen["endpoints"]["dx"]["decision"],
                "plan": chosen["endpoints"]["plan"]["decision"],
                "production_rollout_allowed": chosen["production_rollout"]["allowed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
