from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_mo_calibration_blind_judge import (
    FORBIDDEN_PROMPT_KEYS,
    _select_smoke_rows,
    assert_gce_live_contour,
    audit_prompt_input,
    blind_case_pack,
    build_dx_prompt,
    build_plan_prompt,
    judge_case,
    pin_dx_semantics,
    pin_plan_route,
)


def _source_row() -> dict:
    return {
        "case_id": "real-case-id-must-not-leak",
        "visit_id": "real-visit-id-must-not-leak",
        "overall_pct": "SCORE_CANARY_91827",
        "reg55_section_pct": "REG55_CANARY_77261",
        "attention_reason_ru": "QUEUE_CANARY_66319",
        "findings": [{"evidence": "FINDING_CANARY_55103"}],
        "protocol_suggest": {"items": []},
        "clinical": {
            "age_years": 55,
            "sex": "female",
            "doctor_specialization": "Терапевт",
            "complaints": "головная боль",
            "anamnesis": "повышение давления",
            "objective_status": "АД повышено",
            "exam_data": "данные обследования",
            "clinical_diagnosis": "артериальная гипертензия",
            "mkb_code_main": "I10",
            "exam_recommendations": "контроль АД",
            "treatment_recommendations": "назначение указано",
            "follow_up": "повторный приём",
        },
    }


def test_blind_pack_is_allowlist_and_drops_engine_outputs() -> None:
    pack = blind_case_pack(_source_row(), sample_id="S001")
    serialized = json.dumps(pack, ensure_ascii=False)
    assert pack["sample_id"] == "S001"
    assert "real-case-id-must-not-leak" not in serialized
    assert "SCORE_CANARY_91827" not in serialized
    assert "REG55_CANARY_77261" not in serialized
    assert "QUEUE_CANARY_66319" not in serialized
    assert not any(key in serialized for key in FORBIDDEN_PROMPT_KEYS)


def test_dx_prompt_cannot_see_plan_or_engine_canaries() -> None:
    row = _source_row()
    pack = blind_case_pack(row, sample_id="S001")
    prompt, prompt_input = build_dx_prompt(pack)
    assert "контроль АД" not in prompt
    assert "назначение указано" not in prompt
    assert "повторный приём" not in prompt
    assert "SCORE_CANARY_91827" not in prompt
    assert "REG55_CANARY_77261" not in prompt
    assert audit_prompt_input(prompt_input, source_row=row)["passed"] is True


def test_plan_prompt_accepts_diagnosis_as_premise_without_stage_a_result() -> None:
    pack = blind_case_pack(_source_row(), sample_id="S001")
    prompt, prompt_input = build_plan_prompt(pack, route="llm_no_kp")
    assert "accepted_diagnosis" in prompt
    assert "головная боль" not in prompt
    assert "dx_evidence_pct" not in prompt
    assert "diagnosis_score" not in prompt
    assert "plan_general_llm_pct" in prompt
    assert "protocol_requirements" not in prompt_input


def test_grounded_prompt_contains_only_supplied_kp_requirements() -> None:
    pack = blind_case_pack(_source_row(), sample_id="S001")
    prompt, prompt_input = build_plan_prompt(
        pack,
        route="kp_grounded",
        protocol_context={
            "kp_path": "protocols/example.pdf",
            "kp_trust": "A",
            "required_exams": ["контроль"],
            "treatment": ["группа терапии"],
            "follow_up": ["повторный осмотр"],
            "safety": [],
            "source_refs": ["protocols/example.pdf#p10"],
        },
    )
    assert "protocol_requirements" in prompt_input
    assert "protocols/example.pdf#p10" in prompt
    assert "plan_protocol_pct" in prompt
    assert "plan_general_llm_pct" not in prompt


def test_leakage_audit_rejects_forbidden_fields_and_canaries() -> None:
    audit = audit_prompt_input(
        {"safe": "SCORE_CANARY_91827", "overall_pct": 99},
        source_row=_source_row(),
    )
    assert audit["passed"] is False
    assert "$.overall_pct" in audit["forbidden_paths"]
    assert "overall_pct" in audit["leaked_canaries"]


def test_live_judge_is_hard_blocked_outside_gce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_LLM_EXECUTION_HOST", raising=False)
    monkeypatch.delenv("RUN_HOST", raising=False)
    with pytest.raises(RuntimeError, match="only on GCE"):
        assert_gce_live_contour()


def test_dry_run_builds_two_blind_stages_without_model_call() -> None:
    result = judge_case(
        _source_row(),
        sample_id="S001",
        pass_no=1,
        model="test-model",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["route"] == "llm_no_kp"
    assert result["leakage_audit"]["dx"]["passed"] is True
    assert result["leakage_audit"]["plan"]["passed"] is True
    assert result["prompt_hashes"]["dx"] != result["prompt_hashes"]["plan"]


def test_smoke_selection_covers_kp_and_no_kp_routes(tmp_path: Path) -> None:
    rows = [{"row": index} for index in range(8)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"signals": {"kp_matched": index in {3, 6}}})
            for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    selected = _select_smoke_rows(rows, manifest_path=manifest, limit=5)
    assert [row["row"] for row in selected[:2]] == [3, 6]
    assert len(selected) == 5


def test_controller_pins_grounded_route_fields() -> None:
    pinned = pin_plan_route(
        {
            "provenance": "llm_no_kp",
            "kp_status": "unmatched",
            "plan_protocol_pct": 70,
        },
        route="kp_grounded",
        protocol_context={
            "kp_path": "protocols/example.pdf",
            "kp_trust": "A",
        },
    )
    assert pinned["provenance"] == "kp_grounded"
    assert pinned["kp_status"] == "matched"
    assert pinned["kp_path"] == "protocols/example.pdf"
    assert pinned["kp_trust"] == "A"


def test_controller_downgrades_unsupported_icd_mismatch() -> None:
    pinned = pin_dx_semantics(
        {
            "icd_fit": "mismatch",
            "not_supported_by": [],
            "contradictions": [],
            "provenance": "methodist",
        }
    )
    assert pinned["icd_fit"] == "unknown"
    assert pinned["provenance"] == "llm_blind"
