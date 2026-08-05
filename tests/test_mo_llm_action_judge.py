from __future__ import annotations

import pytest

from clinical_knowledge.mo_llm_action_judge import (
    EXAMPLE_STAGE_A,
    EXAMPLE_STAGE_B,
    build_prompt_a,
    build_prompt_b,
    extract_json_object,
    stage_a_digest,
    validate_stage_a,
    validate_stage_b,
)


def test_validate_examples() -> None:
    a = validate_stage_a(EXAMPLE_STAGE_A)
    b = validate_stage_b(EXAMPLE_STAGE_B)
    assert a["stage"] == "A"
    assert b["stage"] == "B"
    assert a["diagnosis_assessment"]["score_pct"] == 35
    assert b["plan_assessment"]["score_pct"] == 25


def test_extract_json_from_fence() -> None:
    raw = "вот ответ:\n```json\n" + '{"stage":"A","x":1}' + "\n```"
    obj = extract_json_object(raw)
    assert obj["x"] == 1


def test_reject_bad_verdict() -> None:
    bad = dict(EXAMPLE_STAGE_A)
    bad["diagnosis_assessment"] = dict(EXAMPLE_STAGE_A["diagnosis_assessment"])
    bad["diagnosis_assessment"]["verdict"] = "awesome"
    with pytest.raises(ValueError, match="verdict"):
        validate_stage_a(bad)


def test_prompts_contain_case_id() -> None:
    pack = {
        "meta": {"case_id": "3646270", "visit_id": "3646270", "mis_id": "1"},
        "slots": {"complaints": "хромота", "clinical_diagnosis": "M60"},
    }
    a = validate_stage_a(EXAMPLE_STAGE_A)
    pa = build_prompt_a(pack)
    pb = build_prompt_b(pack, stage_a_digest(a))
    assert "3646270" in pa
    assert "stage_a_ref" in pb or "Итог этапа A" in pb
    assert "Этап A" in pa
    assert "Этап B" in pb
