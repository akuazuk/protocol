from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_mo_calibration_llm_methodist_labels import (
    REVIEWER_ID,
    label_from_result,
    run_labels,
    sanitize_blocked_scores,
)


def test_sanitize_blocked_scores_strips_plan_pct() -> None:
    cleaned = sanitize_blocked_scores(
        "plan",
        {
            "verdict": "blocked",
            "plan_protocol_pct": 40,
            "exam_protocol_pct": 10,
            "plan_general_llm_pct": 20,
        },
    )
    assert cleaned["plan_protocol_pct"] is None
    assert cleaned["exam_protocol_pct"] is None
    assert cleaned["plan_general_llm_pct"] is None


def test_label_from_result_maps_dx_and_plan_contracts() -> None:
    dx = label_from_result(
        "dx",
        {
            "verdict": "poor",
            "dx_evidence_pct": 40,
            "potential_harm": True,
            "icd_fit": "mismatch",
            "summary_ru": "Диагноз не подтверждён обследованиями",
        },
    )
    assert dx["score_pct"] == 40
    assert dx["icd_fit"] == "mismatch"
    assert dx["potential_harm"] is True
    plan = label_from_result(
        "plan",
        {
            "verdict": "good",
            "plan_protocol_pct": 88,
            "potential_harm": False,
            "summary_ru": "План соответствует требованиям КП",
        },
    )
    assert plan["icd_fit"] == "na"
    assert plan["score_pct"] == 88


def test_dry_run_fills_all_labels_and_unseals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "mo-score-v3-2026-08-01-2026-08-08"
    review = root / "secret" / "methodist"
    review.mkdir(parents=True)
    cases = []
    labels = []
    pilot_rows = []
    for index in range(1, 16):
        sample_id = f"S{index:03d}"
        cases.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "required_endpoints": ["dx"],
                "clinical_case": {
                    "sample_id": sample_id,
                    "meta": {"specialty": "test"},
                    "evidence": {"complaints": "x"},
                    "diagnosis": {"clinical_diagnosis": "y"},
                    "plan": {"treatment_recommendations": "z"},
                },
                "plan_route": {"route": "llm_no_kp", "kp_status": "unmatched"},
                "protocol_context": None,
                "instructions": {"blind": "x", "dx": "y", "plan": "z"},
            }
        )
        labels.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "endpoint": "dx",
                "verdict": None,
                "score_pct": None,
                "potential_harm": None,
                "icd_fit": None,
                "confidence": None,
                "rationale": "",
                "reviewer_id": "",
                "reviewed_at": "",
            }
        )
        pilot_rows.extend(
            [
                {
                    "kind": "pass",
                    "sample_id": sample_id,
                    "pass_no": 1,
                    "dx_evidence": {"verdict": "poor", "dx_evidence_pct": 35},
                    "plan_concordance": {"verdict": "good", "plan_general_llm_pct": 80},
                },
                {
                    "kind": "pass",
                    "sample_id": sample_id,
                    "pass_no": 2,
                    "dx_evidence": {"verdict": "partial", "dx_evidence_pct": 55},
                    "plan_concordance": {"verdict": "good", "plan_general_llm_pct": 82},
                },
                {
                    "kind": "adjudication",
                    "sample_id": sample_id,
                    "endpoint": "dx",
                    "result": {"verdict": "poor", "dx_evidence_pct": 40},
                },
            ]
        )
    (review / "methodist_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )
    (review / "methodist_labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    (root / "secret" / "blind_pilot.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot_rows),
        encoding="utf-8",
    )
    (root / "methodist_status.json").write_text(
        json.dumps({"schema_version": 1, "comparison_unsealed": False}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MO_CALIBRATION_C6_ROOT", str(root))
    summary = run_labels(model="dry", dry_run=True, limit=0)
    assert summary["written_n"] == 15
    assert summary["error_n"] == 0
    assert summary["passed"] is True
    assert summary["comparison_unsealed"] is True
    assert summary["reviewer_id"] == REVIEWER_ID
    saved = [
        json.loads(line)
        for line in (review / "methodist_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["reviewer_id"] == REVIEWER_ID for row in saved)
    assert all(row["verdict"] == "partial" for row in saved)
