"""Ночной Gemini: лимиты, без thinking, без Pro на needs_human, batch payload."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from clinical_knowledge.gemini_batch import (
    batch_create_body,
    extract_text_from_generate_content_response,
    inlined_request,
    rest_generation_config,
    usage_from_generate_content_response,
)
from clinical_knowledge.gemini_lite import (
    is_terminal_billing_error,
    json_generation_config_dict,
    thinking_budget,
)
from clinical_knowledge.mo_llm_usage import calculate_cost_usd, load_pricing, response_usage
from scripts.grade_kz_llm import _should_escalate, is_retryable_grade_error
from scripts.run_mo_action_queue_llm_judge import (
    document_ready_for_llm,
    document_to_case_pack,
    load_local_case_document,
)


def test_thinking_budget_defaults_to_zero(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    assert thinking_budget() == 0
    cfg = json_generation_config_dict()
    assert cfg["thinking_config"]["thinking_budget"] == 0
    rest = rest_generation_config()
    assert rest["thinkingConfig"]["thinkingBudget"] == 0


def test_needs_human_does_not_escalate_to_pro() -> None:
    do, reason = _should_escalate(
        {"confidence": 0.9, "needs_human": True, "potential_harm": False},
        {"has_potential_harm": False},
    )
    assert do is False
    assert reason == ""


def test_still_escalates_on_parse_and_harm() -> None:
    do, reason = _should_escalate({"_parse_error": "no_json"}, None)
    assert do and reason == "parse_error"
    do, reason = _should_escalate(
        {"confidence": 0.9, "potential_harm": False},
        {"has_potential_harm": True},
    )
    assert do and reason == "harm_disagreement"


def test_spend_cap_is_terminal_and_not_retried() -> None:
    err = "all_llm_models_failed:429 Your project has exceeded its monthly spending cap."
    assert is_terminal_billing_error(err)
    assert is_retryable_grade_error({"_error": err}) is False
    assert is_retryable_grade_error({"_error": "User location is not supported"}) is True


def test_intro_flash_pricing_matches_google() -> None:
    load_pricing.cache_clear()
    assert calculate_cost_usd("gemini-3.6-flash", 1_000_000, 1_000_000) == 4.5


def test_thoughts_tokens_are_billed_as_output() -> None:
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=50,
        )
    )
    assert response_usage(resp) == (100, 70)


def test_batch_body_has_thinking_off_and_json_mime() -> None:
    body = batch_create_body("gemini-3.6-flash", ["p1", "p2"], display_name="t")
    reqs = body["batch"]["inputConfig"]["requests"]["requests"]
    assert len(reqs) == 2
    cfg = reqs[0]["generationConfig"]
    assert cfg["thinkingConfig"]["thinkingBudget"] == 0
    assert cfg["responseMimeType"] == "application/json"
    assert inlined_request("hello")["contents"][0]["parts"][0]["text"] == "hello"


def test_batch_response_parser() -> None:
    payload = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 6,
        },
    }
    assert extract_text_from_generate_content_response(payload) == '{"ok": true}'
    assert usage_from_generate_content_response(payload) == (10, 10)


def test_action_judge_skips_gemini_without_document() -> None:
    pack = document_to_case_pack({"visit_id": "1"}, {"_error": "HTTP 503"})
    assert document_ready_for_llm({"_error": "HTTP 503"}, pack)
    empty = document_to_case_pack({"visit_id": "1"}, {"clinical": {}})
    assert document_ready_for_llm({"ok": True, "clinical": {}}, empty) == "empty_clinical_slots"


def test_action_judge_loads_local_cases_jsonl(tmp_path: Path) -> None:
    day = "2026-09-16"
    y, m, _ = day.split("-")
    cases_dir = tmp_path / "secure_cases" / y / m
    cases_dir.mkdir(parents=True)
    (cases_dir / f"kz_l1_{day}_cases.jsonl").write_text(
        json.dumps(
            {
                "visit_id": "111",
                "mis_id": "m1",
                "complaints": "боль в горле",
                "clinical_diagnosis": "J02.9",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    doc = load_local_case_document({"visit_id": "111"}, day=day, medical_root=tmp_path)
    assert doc.get("ok") is True
    pack = document_to_case_pack({"visit_id": "111"}, doc)
    assert document_ready_for_llm(doc, pack) == ""
    missing = load_local_case_document({"visit_id": "999"}, day=day, medical_root=tmp_path)
    assert missing.get("_error") == "local_case_not_found"
