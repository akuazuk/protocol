#!/usr/bin/env python3
"""Compare score families against completed C6 labels (human or LLM-proxy gold)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.eval_mo_score_agent_proxy import (
    _blind_candidates,
    _ensembles,
    _metric_row,
    _replay_candidates,
    _snapshot_candidates,
    _sha256,
)

BAD_VERDICTS = frozenset({"poor", "critical"})
EXCLUDED_VERDICTS = frozenset({"blocked", "na"})


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _label_reference(labels: list[dict[str, Any]], endpoint: str) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for row in labels:
        if str(row.get("endpoint")) != endpoint:
            continue
        verdict = str(row.get("verdict") or "")
        if verdict in EXCLUDED_VERDICTS:
            continue
        try:
            score = float(row.get("score_pct"))
        except (TypeError, ValueError):
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        references[sample_id] = {
            "score": score,
            "bad": verdict in BAD_VERDICTS or bool(row.get("potential_harm")),
        }
    return references


def evaluate_c7(
    snapshots: list[dict[str, Any]],
    replays: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
    gold_kind: str = "llm_proxy_c6b",
) -> dict[str, Any]:
    snapshot_candidates, mis_to_sample = _snapshot_candidates(snapshots)
    base = {**snapshot_candidates, **_replay_candidates(replays, mis_to_sample)}
    endpoint_reports: dict[str, Any] = {}
    for endpoint in ("dx", "plan"):
        references = _label_reference(labels, endpoint)
        candidates = {**base, **_blind_candidates(blind, endpoint)}
        candidates.update(_ensembles(candidates, endpoint))
        metrics = {
            name: _metric_row(
                references,
                values,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + index * 11,
            )
            for index, (name, values) in enumerate(sorted(candidates.items()))
            if len(set(references) & set(values)) >= 5
        }
        ranking = sorted(
            metrics,
            key=lambda name: (
                metrics[name].get("pr_auc_bad") is not None,
                metrics[name].get("pr_auc_bad") or -1,
                metrics[name].get("roc_auc_bad") or -1,
                -(metrics[name].get("mae") or 10_000),
            ),
            reverse=True,
        )
        endpoint_reports[endpoint] = {
            "gold_labeled_n": len(references),
            "gold_bad_n": sum(bool(value["bad"]) for value in references.values()),
            "candidate_n": len(metrics),
            "ranking_by_gold_pr_auc": ranking,
            "metrics": metrics,
        }
    return {
        "schema_version": 1,
        "analysis": "calibration_c7_against_c6_labels",
        "gold_kind": gold_kind,
        "proxy_is_human_gold": gold_kind == "methodist_human",
        "production_decision_allowed": False,
        "endpoints": endpoint_reports,
        "limitations": {
            "formal_human_methodist_labels": gold_kind == "methodist_human",
            "production_rollout_allowed": False,
            "required_next_gate": (
                "confirmatory_cohort_and_owner_review"
                if gold_kind != "methodist_human"
                else "confirmatory_cohort"
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
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gold-kind", default="llm_proxy_c6b")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = evaluate_c7(
        _rows(args.snapshot),
        _rows(args.replay),
        _rows(args.blind),
        _rows(args.labels),
        bootstrap_iterations=max(0, args.bootstrap_iterations),
        seed=args.seed,
        gold_kind=args.gold_kind,
    )
    report["provenance"] = {
        "evaluator_sha256": _sha256(Path(__file__)),
        "input_sha256": {
            "snapshot": _sha256(args.snapshot),
            "replay": _sha256(args.replay),
            "blind": _sha256(args.blind),
            "labels": _sha256(args.labels),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": report["analysis"],
                "gold_kind": report["gold_kind"],
                "dx": {
                    "labeled": report["endpoints"]["dx"]["gold_labeled_n"],
                    "bad": report["endpoints"]["dx"]["gold_bad_n"],
                    "top": (report["endpoints"]["dx"]["ranking_by_gold_pr_auc"] or [None])[0],
                },
                "plan": {
                    "labeled": report["endpoints"]["plan"]["gold_labeled_n"],
                    "bad": report["endpoints"]["plan"]["gold_bad_n"],
                    "top": (report["endpoints"]["plan"]["ranking_by_gold_pr_auc"] or [None])[0],
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
