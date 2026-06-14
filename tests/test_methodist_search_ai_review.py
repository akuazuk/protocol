"""Тесты AI-оценки выдачи поиска для кабинета методиста."""
from __future__ import annotations

import pytest

from clinical_knowledge.methodist_search_ai_review import (
    normalize_search_ai_review,
    _build_prompt,
)


def test_normalize_search_ai_review_correct_without_fix():
    raw = {
        "ranking_verdict": "correct",
        "ranking_rating": 5,
        "summary_ru": "Top-1 релевантен.",
        "engine_improvements_ru": [],
        "tags": [],
        "top1_relevant": True,
        "retrieval_fix": None,
        "confidence": "high",
    }
    out = normalize_search_ai_review(raw)
    assert out["ranking_verdict"] == "correct"
    assert out["ranking_rating"] == 5
    assert out["retrieval_fix"] is None
    assert out["review_source"] == "ai_assisted"


def test_normalize_search_ai_review_with_retrieval_fix():
    raw = {
        "ranking_verdict": "wrong",
        "ranking_rating": 2,
        "summary_ru": "Неверный top.",
        "engine_improvements_ru": ["Усилить routing по МКБ I10"],
        "tags": ["wrong_protocol"],
        "retrieval_fix": {
            "rejected_path": "cardio/wrong.pdf",
            "chosen_path": "cardio/right.pdf",
            "note": "test",
        },
        "confidence": "medium",
    }
    out = normalize_search_ai_review(raw)
    assert out["retrieval_fix"]["chosen_path"] == "cardio/right.pdf"
    assert out["tags"] == ["wrong_protocol"]


def test_normalize_rejects_invalid_verdict():
    with pytest.raises(ValueError, match="ranking_verdict"):
        normalize_search_ai_review({"ranking_verdict": "bad", "ranking_rating": 3})


def test_build_prompt_includes_query_and_paths():
    prompt = _build_prompt(
        {
            "query": "I10 гипертензия",
            "llm_json": {
                "protocols": [
                    {"path": "bolezni/foo.pdf", "confidence_score": 0.82, "match_reason": "МКБ I10"},
                ],
            },
            "retrieval": [{"path": "bolezni/foo.pdf", "score": 12.3}],
            "icd_codes": ["I10"],
        }
    )
    assert "I10 гипертензия" in prompt
    assert "foo.pdf" in prompt
