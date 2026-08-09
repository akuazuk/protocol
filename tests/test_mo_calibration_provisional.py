from __future__ import annotations

from scripts.select_mo_calibration_provisional import choose_provisional


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
