from __future__ import annotations

import pytest

from clinical_knowledge.mo_daily import sanitize_mo_org_label
from clinical_knowledge.mo_llm_action_judge import (
    EXAMPLE_STAGE_A,
    EXAMPLE_STAGE_B,
    build_prompt_a,
    build_prompt_b,
    extract_json_object,
    load_llm_action_judge_for_case,
    stage_a_digest,
    summarize_llm_action_judge_for_ui,
    validate_stage_a,
    validate_stage_b,
)


def test_validate_examples() -> None:
    a = validate_stage_a(EXAMPLE_STAGE_A)
    b = validate_stage_b(EXAMPLE_STAGE_B)
    assert a["stage"] == "A"
    assert b["stage"] == "B"
    assert a["completeness"]["score_pct"] == 70
    assert a["completeness"]["missing_blocks"] == ["exam_data"]
    assert a["diagnosis_assessment"]["score_pct"] == 35
    assert a["diagnosis_assessment"]["blocked_by_incomplete"] is False
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


def test_verdict_alias_strong_to_good() -> None:
    raw = dict(EXAMPLE_STAGE_A)
    raw["diagnosis_assessment"] = dict(EXAMPLE_STAGE_A["diagnosis_assessment"])
    raw["diagnosis_assessment"]["verdict"] = "strong"
    a = validate_stage_a(raw)
    assert a["diagnosis_assessment"]["verdict"] == "good"


def test_reject_missing_completeness() -> None:
    bad = dict(EXAMPLE_STAGE_A)
    bad.pop("completeness")
    with pytest.raises(ValueError, match="completeness"):
        validate_stage_a(bad)


def test_stage_a_digest_includes_completeness() -> None:
    a = validate_stage_a(EXAMPLE_STAGE_A)
    digest = stage_a_digest(a)
    assert digest["completeness_score_pct"] == 70
    assert "missing:exam_data" in digest["key_gaps"] or "exam_data" in digest["missing_blocks"]


def test_prompts_contain_case_id() -> None:
    pack = {
        "meta": {"case_id": "3646270", "visit_id": "3646270", "mis_id": "1"},
        "slots": {"complaints": "хромота", "clinical_diagnosis": "M60"},
    }
    a = validate_stage_a(EXAMPLE_STAGE_A)
    pa = build_prompt_a(pack)
    pb = build_prompt_b(pack, stage_a_digest(a))
    assert "3646270" in pa
    assert "completeness" in pa
    assert "полнота" in pa.lower() or "Полнота" in pa
    assert "stage_a_ref" in pb or "Итог этапа A" in pb
    assert "Этап A" in pa
    assert "Этап B" in pb


def test_sanitize_mo_org_label_strips_versions() -> None:
    assert sanitize_mo_org_label("v4.0.0", scorer_version="v4.0.0") == ""
    assert sanitize_mo_org_label("4.0", schema_version="4.0") == ""
    assert sanitize_mo_org_label("deep-v2-fallback") == ""
    assert sanitize_mo_org_label("Урология", scorer_version="v4.0.0") == "Урология"


def test_summarize_ui_payload_has_three_kpis() -> None:
    a = validate_stage_a(EXAMPLE_STAGE_A)
    b = validate_stage_b(EXAMPLE_STAGE_B)
    ui = summarize_llm_action_judge_for_ui(
        {
            "case_id": "3646270",
            "date": "2026-08-04",
            "model_a": "gemini-3.6-flash",
            "model_b": "gemini-3.6-flash",
            "stage_a": a,
            "stage_b": b,
        }
    )
    assert ui["available"] is True
    assert ui["shadow"] is True
    assert ui["kpis"]["completeness"]["score_pct"] == 70
    assert ui["kpis"]["diagnosis"]["score_pct"] == 35
    assert ui["kpis"]["recommendations"]["score_pct"] == 25


def test_load_llm_action_judge_from_jsonl(tmp_path) -> None:
    day = "2026-08-04"
    a = validate_stage_a(EXAMPLE_STAGE_A)
    b = validate_stage_b(EXAMPLE_STAGE_B)
    root = tmp_path / "medical_exams"
    path = root / "llm_action_judge" / "2026" / "08" / "04" / "judges.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        __import__("json").dumps(
            {
                "case_id": "3646270",
                "visit_id": "3646270",
                "mis_id": "898517",
                "date": day,
                "stage_a": a,
                "stage_b": b,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ui = load_llm_action_judge_for_case("3646270", visit_date=day, roots=[root])
    assert ui["available"] is True
    assert ui["kpis"]["diagnosis"]["verdict"] == "poor"
    missing = load_llm_action_judge_for_case("no-such", visit_date=day, roots=[root])
    assert missing["available"] is False
