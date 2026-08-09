#!/usr/bin/env python3
"""Choose an exploratory provisional methodology from proxy aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def choose_provisional(report: Mapping[str, Any]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for endpoint in ("dx", "plan"):
        payload = report.get("endpoints", {}).get(endpoint) or {}
        ranking = list(payload.get("ranking_by_proxy_pr_auc") or [])
        metrics = payload.get("metrics") or {}
        top_name = ranking[0] if ranking else None
        top = metrics.get(top_name) if top_name else None
        bad_n = int(payload.get("proxy_bad_n") or 0)
        labeled_n = int(payload.get("proxy_labeled_n") or 0)
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
        "analysis": "agent_proxy_provisional_c8a",
        "proxy_models": list(report.get("proxy_models") or []),
        "endpoints": endpoints,
        "production_rollout": {
            "allowed": False,
            "reason": "proxy_not_human_gold",
            "required_next_gate": "independent_methodist_labels",
        },
        "shadow_recommendation": {
            "dx": endpoints["dx"]["decision"],
            "plan": endpoints["plan"]["decision"],
            "note": (
                "Shadow only. Do not change action queue, thresholds, SSOT, or "
                "warehouse until formal C6-C9 with methodist gold."
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
    parser.add_argument("--proxy-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.proxy_eval.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("proxy eval must be a JSON object")
    chosen = choose_provisional(report)
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
