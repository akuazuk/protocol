"""Тесты AI-оценки для кабинета методиста."""
from __future__ import annotations

import pytest

from clinical_knowledge.methodist_ai_review import (
    methodist_ai_review_enabled,
    normalize_ai_review,
    run_methodist_ai_review,
)


def test_normalize_ai_review_valid():
    raw = {
        "kz_compliance_gold": "mostly_compliant",
        "system_accuracy_rating": 4,
        "system_accuracy_verdict": "mostly_correct",
        "tags": ["false_positive_rule"],
        "summary_ru": "КЗ в целом соответствует.",
        "engine_improvements_ru": ["Отключить ложное правило bladder_dysfunction для M54.1", "Не занижать treatment_score при НПВС"],
        "kz_text_notes_ru": ["Опечатка реблакса"],
        "system_notes_ru": "Правило ЭГДС — ложное срабатывание.",
        "block_overrides": [
            {"block_key": "diagnosis_score", "verdict": "agree", "note": ""},
            {"block_key": "treatment_score", "verdict": "disagree", "note": "Занижено"},
        ],
        "rule_overrides": [{"rule_id": "gerd_required_exam", "human_pass": True, "note": "OK"}],
        "retrieval_fix": {"rejected_path": "a.pdf", "chosen_path": "b.pdf", "note": ""},
        "confidence": "high",
    }
    out = normalize_ai_review(raw)
    assert out["kz_compliance_gold"] == "mostly_compliant"
    assert out["system_accuracy_rating"] == 4
    assert len(out["block_overrides"]) == 1
    assert out["block_overrides"][0]["block_key"] == "treatment_score"
    assert len(out["engine_improvements_ru"]) == 2
    assert out["kz_text_notes_ru"] == ["Опечатка реблакса"]
    assert out["retrieval_fix"]["chosen_path"] == "b.pdf"


def test_normalize_ai_review_rejects_bad_gold():
    with pytest.raises(ValueError, match="kz_compliance_gold"):
        normalize_ai_review({"system_accuracy_rating": 3, "system_accuracy_verdict": "wrong"})


def test_run_methodist_ai_review_mock():
    payload = {
        "kz_compliance_gold": "partially_compliant",
        "system_accuracy_rating": 3,
        "system_accuracy_verdict": "partially_wrong",
        "tags": ["score_misleading"],
        "summary_ru": "Итог",
        "engine_improvements_ru": ["Исправить hybrid gate для rules=0%"],
        "system_notes_ru": "Hybrid завышен",
        "block_overrides": [],
        "rule_overrides": [],
        "confidence": "medium",
    }

    class FakeResp:
        pass

    result = {
        "analysis_id": "test-id",
        "structured_analysis": {"compliance": {"overall_score": 70, "score_breakdown": {}}},
        "clinical_rules": {"rules_check": {"findings": []}},
    }

    out = run_methodist_ai_review(
        result,
        "Диагноз: ГЭРБ K21.9\nЖалобы: изжога.",
        generate_fn=lambda m, p: FakeResp(),
        extract_text_fn=lambda r: __import__("json").dumps(payload),
        parse_json_fn=lambda t: __import__("json").loads(t),
        get_model_fn=lambda: object(),
    )
    assert out["system_accuracy_verdict"] == "partially_wrong"
    assert out["review_source"] == "ai_assisted"


def test_methodist_ai_review_enabled_default():
    assert methodist_ai_review_enabled() is True
