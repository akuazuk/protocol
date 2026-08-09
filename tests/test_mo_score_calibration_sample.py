from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_mo_score_calibration_sample import (
    DEFAULT_SENTINEL,
    arm_d_fingerprint,
    clamp_requirements_to_pool,
    extract_engine_snapshot,
    load_exclude_keys,
    normalize_candidate,
    select_sample,
    write_outputs,
)


def _candidate(index: int) -> dict:
    bands = (
        (20, "0-49"),
        (55, "50-59"),
        (65, "60-69"),
        (75, "70-79"),
        (90, "80+"),
    )
    overall, band = bands[index % len(bands)]
    return {
        "case_key": DEFAULT_SENTINEL if index == 0 else f"case-{index:03d}",
        "visit_date": f"2026-08-0{1 + (index % 8)}",
        "overall_pct": overall,
        "band": band,
        "specialty": f"specialty-{index % 6}",
        "doctor_key": f"doctor-{index // 3}",
        "training_use": index in {1, 2},
        "high_action": overall >= 80 and index < 25,
        "has_p0p1": index < 12,
        "action": index < 25,
        "reg55_pct": 85 if index < 12 else 60,
        "regulatory_pct": 70 if index < 10 else 60,
        "reg55_gap": index < 10,
        "reg55_high_weak": index < 8,
        "icd_dx_dispute": index < 10,
        "kp_matched": index % 2 == 0,
        "kp_trust": "A" if index % 2 == 0 else "",
        "kp_checked": True,
        "has_exam_results": index < 16,
        "has_treatment": index < 16,
        "axes": {
            "documentation": 80,
            "clinical_concordance": 70,
            "safety": 90,
            "regulatory": 70 if index < 10 else 60,
        },
        "warehouse": {
            "overall_pct": overall,
            "overall_pct_v3": overall - 2,
            "rubric_pct": 75,
            "reg55_section_pct": 85 if index < 12 else 60,
            "reg55_band": "compliant_measures",
            "reg55_applicable_n": 6,
            "reg55_weak_points_json": '["criterion"]',
            "scorer_version": "v4.0.0",
            "score_schema_version": "4.0",
        },
        "row": {
            "case_id": DEFAULT_SENTINEL if index == 0 else f"case-{index:03d}",
            "visit_id": f"visit-{index:03d}",
            "mis_id": f"mis-{index:03d}",
            "date": f"2026-08-0{1 + (index % 8)}",
            "overall_pct": overall,
            "doctor_specialization": f"specialty-{index % 6}",
            "clinical": {
                "complaints": "sensitive complaint",
                "clinical_diagnosis": "sensitive diagnosis",
                "exam_data": "sensitive result",
                "treatment_recommendations": "sensitive treatment",
            },
            "deep": {
                "overall_pct": overall - 2,
                "axes": {
                    "documentation": 80,
                    "clinical_concordance": 70,
                    "safety": 90,
                    "regulatory": 70 if index < 10 else 60,
                },
                "findings": [
                    {
                        "code": "P1_TEST",
                        "severity": "P1",
                        "passed": False,
                    }
                ],
            },
        },
    }


def test_select_sample_meets_preregistered_c0_coverage() -> None:
    selected, audit = select_sample([_candidate(index) for index in range(80)])
    assert len(selected) == 30
    assert audit["passed"] is True
    assert audit["deficits"] == {}
    assert audit["sentinel_present"] is True
    assert audit["all_training_use_present"] is True
    assert audit["max_cases_per_doctor"] <= 3
    assert audit["coverage"]["specialties"] >= 4
    assert all(audit["coverage"][f"band:{band}"] >= 4 for band in ("0-49", "50-59", "60-69", "70-79", "80+"))


def test_select_sample_fails_if_sentinel_is_missing() -> None:
    with pytest.raises(ValueError, match="sentinel"):
        select_sample([_candidate(index) for index in range(1, 40)])


def test_select_sample_allows_independent_cohort_without_sentinel() -> None:
    selected, audit = select_sample(
        [_candidate(index) for index in range(1, 80)],
        sentinel="",
    )
    assert len(selected) == 30
    assert audit["passed"] is True
    assert audit["sentinel_required"] is False
    assert audit["sentinel_present"] is True


def test_select_sample_scales_coverage_for_confirmatory_target() -> None:
    pool = []
    for index in range(400):
        item = _candidate(index)
        # Enrich rare strata so the scaled confirmatory floors remain feasible.
        item["high_action"] = index < 40
        item["reg55_gap"] = index < 40
        item["reg55_high_weak"] = index < 30
        item["icd_dx_dispute"] = index < 40
        item["has_exam_results"] = index < 80
        item["has_treatment"] = index < 80
        item["kp_matched"] = index % 2 == 0
        item["kp_checked"] = True
        pool.append(item)
    selected, audit = select_sample(pool, target=100, seed=43, sentinel="")
    assert len(selected) == 100
    assert audit["passed"] is True
    assert audit["deficits"] == {}
    assert audit["max_cases_per_doctor"] <= 3
    assert all(
        audit["coverage"][f"band:{band}"] >= 8
        for band in ("0-49", "50-59", "60-69", "70-79", "80+")
    )


def test_clamp_requirements_drops_unavailable_reg55_floors() -> None:
    pool = [_candidate(index) for index in range(40)]
    for item in pool:
        item["reg55_gap"] = False
        item["reg55_high_weak"] = False
    clamped = clamp_requirements_to_pool(
        {
            "band:0-49": 4,
            "reg55_gap": 6,
            "reg55_high_weak": 4,
            "specialties": 4,
        },
        pool,
    )
    assert clamped["reg55_gap"] == 0
    assert clamped["reg55_high_weak"] == 0
    assert clamped["specialties"] >= 1


def test_load_exclude_keys_reads_manifest_and_blocks_overlap(tmp_path: Path) -> None:
    manifest = tmp_path / "secret_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"case_key": f"case-{index:03d}", "aliases": [f"alias-{index}"]})
            for index in (1, 2, 3)
        )
        + "\n",
        encoding="utf-8",
    )
    exclude = load_exclude_keys(manifest)
    assert exclude == {"case-001", "alias-1", "case-002", "alias-2", "case-003", "alias-3"}
    pool = [_candidate(index) for index in range(80)]
    for item in pool:
        item["aliases"] = [item["case_key"]]
    filtered = [
        item
        for item in pool
        if item["case_key"] not in exclude
        and not set(item.get("aliases") or []) & exclude
    ]
    selected, audit = select_sample(filtered, sentinel="")
    assert audit["passed"] is True
    assert {item["case_key"] for item in selected}.isdisjoint(exclude)


def test_normalize_candidate_extracts_calibration_signals() -> None:
    row = {
        "case_id": "case-1",
        "date": "2026-08-03",
        "overall_pct": 91,
        "doctor_specialization": "Кардиолог",
        "doctor_id": "doctor-1",
        "clinical": {
            "exam_results": "ЭКГ выполнена",
            "treatment_recommendations": "терапия назначена",
        },
        "deep": {
            "axes": {"regulatory": 70},
            "findings": [{"code": "x", "severity": "P1", "passed": False}],
        },
        "reg55_section_pct": 82,
        "reg55_weak_points": ["criterion"],
        "icd_review": {"verdict": "mismatch"},
        "protocol_suggest": {
            "items": [{"match_kind": "clinical", "score": 75, "trust": "A"}]
        },
    }
    item = normalize_candidate(row)
    assert item is not None
    assert item["band"] == "80+"
    assert item["high_action"] is True
    assert item["reg55_gap"] is True
    assert item["reg55_high_weak"] is True
    assert item["icd_dx_dispute"] is True
    assert item["kp_matched"] is True
    assert item["has_exam_results"] is True
    assert item["has_treatment"] is True


def test_engine_snapshot_retains_every_existing_score_family() -> None:
    snapshot = extract_engine_snapshot(_candidate(0))
    scores = snapshot["scores"]
    assert scores["overall_pct"] == 20
    assert scores["overall_pct_v3"] == 18
    assert set(scores["axes"]) == {
        "documentation",
        "clinical_concordance",
        "safety",
        "regulatory",
    }
    assert set(scores["zones"]) == {"zone1", "zone2a", "zone2b"}
    assert scores["rubric_pct"] == 75
    assert scores["reg55"]["score_pct"] == 85
    assert snapshot["findings"]["n_by_severity"]["P1"] == 1
    assert "action" in snapshot
    assert "icd_pipeline" in snapshot
    assert "existing_llm" in snapshot
    assert snapshot["snapshot_hash"]


def test_public_manifest_does_not_contain_secret_values(tmp_path: Path) -> None:
    selected, audit = select_sample([_candidate(index) for index in range(80)])
    public_path = tmp_path / "public" / "manifest.json"
    public = write_outputs(
        selected,
        audit,
        secret_dir=tmp_path / "secret",
        public_manifest=public_path,
        source_paths=[Path("/var/data/medical_exams/secure_cases/input.jsonl")],
        seed=42,
        replay_rows=[],
        replay_audit={"attempted_n": 0, "all_cases_reproducible": False},
    )
    serialized = public_path.read_text(encoding="utf-8")
    assert public["audit"]["passed"] is True
    assert DEFAULT_SENTINEL not in serialized
    assert "doctor-" not in serialized
    assert "specialty-" not in serialized
    assert "sensitive complaint" not in serialized
    assert "sensitive diagnosis" not in serialized
    assert len((tmp_path / "secret" / "secret_cases.jsonl").read_text(encoding="utf-8").splitlines()) == 30
    assert json.loads(serialized)["phi_check"]["contains_clinical_text"] is False


def test_public_manifest_cannot_be_written_under_secret_dir(tmp_path: Path) -> None:
    selected, audit = select_sample([_candidate(index) for index in range(80)])
    with pytest.raises(ValueError, match="outside secret-dir"):
        write_outputs(
            selected,
            audit,
            secret_dir=tmp_path / "secret",
            public_manifest=tmp_path / "secret" / "manifest.json",
            source_paths=[],
            seed=42,
        )


def test_arm_d_fingerprint_freezes_code_config_and_protocol_summaries() -> None:
    first = arm_d_fingerprint()
    second = arm_d_fingerprint()
    assert first == second
    assert len(first["fingerprint"]) == 64
    assert first["component_hashes"]["config/mo_scorer_v4.yaml"] != "missing"
    assert "protocol_summary_tree_hash" in first
    assert "environment_flags" in first
