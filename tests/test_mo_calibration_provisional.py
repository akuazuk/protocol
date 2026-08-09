from __future__ import annotations

from scripts.select_mo_calibration_provisional import choose_provisional


def test_c7_llm_proxy_gold_is_unstable_with_tiny_bad_n() -> None:
    report = {
        "gold_kind": "llm_proxy_c6b",
        "endpoints": {
            "dx": {
                "gold_labeled_n": 7,
                "gold_bad_n": 1,
                "ranking_by_gold_pr_auc": ["blind.mean_2"],
                "metrics": {
                    "blind.mean_2": {
                        "n": 7,
                        "proxy_bad_n": 1,
                        "mae": 18.7,
                        "roc_auc_bad": 1.0,
                        "pr_auc_bad": 1.0,
                    }
                },
            },
            "plan": {
                "gold_labeled_n": 7,
                "gold_bad_n": 2,
                "ranking_by_gold_pr_auc": ["snapshot.overall_v3"],
                "metrics": {
                    "snapshot.overall_v3": {
                        "n": 7,
                        "proxy_bad_n": 2,
                        "mae": 17.1,
                        "roc_auc_bad": 0.85,
                        "pr_auc_bad": 0.75,
                    }
                },
            },
        },
    }
    chosen = choose_provisional(report)
    assert chosen["analysis"] == "c8b_c7_llm_proxy_gold_provisional"
    assert chosen["production_rollout"]["allowed"] is False
    assert chosen["endpoints"]["dx"]["decision"] == "no_stable_provisional"
    assert chosen["endpoints"]["plan"]["decision"] == "no_stable_provisional"


def test_provisional_blocks_production_and_requires_stable_plan() -> None:
    report = {
        "proxy_models": ["gemini-3.1-pro-preview"],
        "endpoints": {
            "dx": {
                "proxy_labeled_n": 25,
                "proxy_bad_n": 2,
                "ranking_by_proxy_pr_auc": ["blind.adjudicated_or_mean"],
                "metrics": {
                    "blind.adjudicated_or_mean": {
                        "n": 23,
                        "proxy_bad_n": 2,
                        "mae": 20.6,
                        "spearman": -0.05,
                        "roc_auc_bad": 0.61,
                        "pr_auc_bad": 0.23,
                        "pr_auc_bad_ci95": [0.05, 1.0],
                        "classification_at_55": {"sensitivity": 0.5},
                    }
                },
            },
            "plan": {
                "proxy_labeled_n": 21,
                "proxy_bad_n": 8,
                "ranking_by_proxy_pr_auc": ["blind.pass_1", "blind.mean_2"],
                "metrics": {
                    "blind.pass_1": {
                        "n": 20,
                        "proxy_bad_n": 8,
                        "mae": 25.95,
                        "spearman": 0.48,
                        "roc_auc_bad": 0.64,
                        "pr_auc_bad": 0.59,
                        "pr_auc_bad_ci95": [0.26, 0.86],
                        "classification_at_55": {"sensitivity": 0.75},
                    }
                },
            },
        },
    }
    chosen = choose_provisional(report)
    assert chosen["production_rollout"]["allowed"] is False
    assert chosen["endpoints"]["dx"]["decision"] == "no_stable_provisional"
    assert chosen["endpoints"]["plan"]["decision"] == "provisional_shadow:blind.pass_1"
    assert chosen["shadow_recommendation"]["plan"] == "provisional_shadow:blind.pass_1"
