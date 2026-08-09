from __future__ import annotations

import json

from scripts.eval_mo_score_agent_proxy import evaluate_agent_proxy


def _fixtures() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    snapshots = []
    replays = []
    blind = []
    proxy = []
    for index in range(20):
        sample_id = f"S{index + 1:03d}"
        mis_id = str(9000 + index)
        proxy_score = 20.0 if index < 5 else 90.0
        verdict = "poor" if index < 5 else "good"
        snapshots.append(
            {
                "sample_id": str(10000 + index),
                "source_ids": {"mis_id": mis_id},
                "scores": {
                    "overall_pct": 100 - proxy_score,
                    "overall_pct_v3": 50,
                    "rubric_pct": 60,
                    "reg55": {"score_pct": 70},
                    "axes": {
                        "clinical_concordance": proxy_score,
                        "documentation": 80,
                        "regulatory": 70,
                        "safety": 90,
                    },
                    "zones": {
                        "zone1": {"score_pct": 80},
                        "zone2a": {"score_pct": proxy_score},
                        "zone2b": {"score_pct": proxy_score},
                    },
                },
            }
        )
        replays.append(
            {
                "case_key": mis_id,
                "comparisons": {
                    "overall_pct": {"replayed": 55},
                    "axis:clinical_concordance": {"replayed": proxy_score},
                    "axis:documentation": {"replayed": 80},
                    "axis:regulatory": {"replayed": 70},
                    "axis:safety": {"replayed": 90},
                },
            }
        )
        for pass_no, delta in ((1, 2), (2, -2)):
            blind.append(
                {
                    "kind": "pass",
                    "sample_id": sample_id,
                    "pass_no": pass_no,
                    "error": None,
                    "dx_evidence": {
                        "dx_evidence_pct": proxy_score + delta,
                        "verdict": verdict,
                        "potential_harm": False,
                    },
                    "plan_concordance": {
                        "plan_protocol_pct": proxy_score + delta,
                        "verdict": verdict,
                        "potential_harm": False,
                    },
                }
            )
        proxy.append(
            {
                "kind": "pass",
                "sample_id": sample_id,
                "pass_no": 1,
                "model": "gemini-3.1-pro-preview",
                "error": None,
                "dx_evidence": {
                    "dx_evidence_pct": proxy_score,
                    "verdict": verdict,
                    "potential_harm": False,
                },
                "plan_concordance": {
                    "plan_protocol_pct": proxy_score,
                    "verdict": verdict,
                    "potential_harm": False,
                },
            }
        )
    return snapshots, replays, blind, proxy


def test_proxy_analysis_ranks_relevant_scores_without_claiming_gold() -> None:
    report = evaluate_agent_proxy(*_fixtures(), bootstrap_iterations=100, seed=7)
    assert report["analysis"] == "agent_proxy_exploratory_c7a"
    assert report["proxy_models"] == ["gemini-3.1-pro-preview"]
    assert report["limitations"]["proxy_is_human_gold"] is False
    assert report["limitations"]["production_decision_allowed"] is False
    assert report["endpoints"]["dx"]["proxy_labeled_n"] == 20
    assert report["endpoints"]["dx"]["proxy_bad_n"] == 5
    perfect = report["endpoints"]["dx"]["metrics"]["snapshot.zone2a"]
    assert perfect["roc_auc_bad"] == 1.0
    assert perfect["pr_auc_bad"] == 1.0
    assert perfect["classification_at_55"]["balanced_accuracy"] == 1.0
    assert perfect["mae"] == 0
    assert perfect["mae_ci95"] == [0.0, 0.0]
    assert "snapshot.zone2a" in report["endpoints"]["dx"]["ranking_by_proxy_pr_auc"]
    assert "snapshot.zone1" not in report["endpoints"]["dx"]["ranking_by_proxy_pr_auc"]
    assert "snapshot.zone1" in report["endpoints"]["dx"]["control_metrics"]
    assert "snapshot.zone2b" in report["endpoints"]["plan"]["ranking_by_proxy_pr_auc"]


def test_proxy_report_is_aggregate_and_phi_safe() -> None:
    snapshots, replays, blind, proxy = _fixtures()
    proxy[0]["dx_evidence"] = {
        "dx_evidence_pct": None,
        "verdict": "blocked",
        "potential_harm": False,
    }
    report = evaluate_agent_proxy(
        snapshots,
        replays,
        blind,
        proxy,
        bootstrap_iterations=0,
    )
    assert report["endpoints"]["dx"]["proxy_labeled_n"] == 19
    serialized = json.dumps(report, ensure_ascii=False)
    assert "S001" not in serialized
    assert "9000" not in serialized
    assert report["phi_check"] == {
        "contains_source_identifiers": False,
        "contains_sample_identifiers": False,
        "contains_clinical_text": False,
    }


def test_valid_endpoint_survives_other_endpoint_contract_error() -> None:
    snapshots, replays, blind, proxy = _fixtures()
    proxy[0]["error"] = "ValueError: blocked plan must not have scores"
    proxy[0]["plan_concordance"] = None
    report = evaluate_agent_proxy(
        snapshots,
        replays,
        blind,
        proxy,
        bootstrap_iterations=0,
    )
    assert report["proxy_run_quality"]["error_row_n"] == 1
    assert report["proxy_run_quality"]["error_class_counts"] == {"ValueError": 1}
    assert report["endpoints"]["dx"]["proxy_labeled_n"] == 20
    assert report["endpoints"]["plan"]["proxy_labeled_n"] == 19
    assert report["endpoints"]["plan"]["proxy_abstention_n"] == 0
