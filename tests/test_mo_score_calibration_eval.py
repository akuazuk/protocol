from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_mo_score_calibration import replay_drift


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_replay_drift_report_is_aggregate_and_phi_safe(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    snapshot = tmp_path / "snapshot.jsonl"
    replay = tmp_path / "replay.jsonl"
    _write(
        cases,
        {
            "case_id": "secret-case",
            "_content_hash": "source-hash",
            "overall_pct": 80,
            "deep": {"overall_pct": 75},
            "evaluation_v4": {
                "score_pct": 80,
                "scorer_version": "v4.0.0",
                "axes": {"documentation": 90},
            },
        },
    )
    _write(
        snapshot,
        {
            "source_ids": {"case_id": "secret-case"},
            "scores": {
                "overall_pct": 70,
                "axes": {"documentation": 90},
            },
            "versions": {"content_hash": "warehouse-hash"},
        },
    )
    _write(
        replay,
        {
            "arm_d_fingerprint": "fingerprint",
            "comparisons": {
                "overall_pct": {
                    "comparable": True,
                    "match": False,
                    "delta": 10,
                }
            },
        },
    )
    report = replay_drift(cases, snapshot, replay)
    serialized = json.dumps(report)
    assert report["case_n"] == 1
    assert report["replay_mismatch_counts"]["overall_pct"] == 1
    assert report["conclusion"] == "stored_snapshot_not_replay_baseline"
    assert report["phi_check"]["contains_row_identifiers"] is False
    assert "secret-case" not in serialized
