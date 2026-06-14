"""Детерминированная оценка выдачи поиска (fallback без LLM)."""
from __future__ import annotations

from clinical_knowledge.methodist_search_ai_review import build_deterministic_search_ai_review


def test_deterministic_flags_symptom_only_cough_fever():
    out = build_deterministic_search_ai_review(
        {
            "query": "кашель и температура 39",
            "llm_json": {
                "protocols": [
                    {
                        "path": "pulmonologiya/кп_микобактериоз.pdf",
                        "title": "Диагностика лечение микобактериоза",
                        "confidence_score": 0.94,
                    }
                ],
            },
            "retrieval": [],
            "icd_codes": [],
            "retrieve_only": True,
        }
    )
    assert out["review_source"] == "deterministic_fallback"
    assert "query_too_vague" in out["tags"]
    assert out["top1_relevant"] is False
    assert out.get("suggested_funnel_step") in (0, 2, 4)
    assert out["retrieval_fix"] is not None
    assert "микобактер" in out["retrieval_fix"]["rejected_path"].lower() or out["retrieval_fix"][
        "rejected_path"
    ]


def test_deterministic_ok_with_icd():
    out = build_deterministic_search_ai_review(
        {
            "query": "J18.9 пневмония",
            "llm_json": {
                "protocols": [
                    {"path": "pulmonologiya/pneumonia.pdf", "confidence_score": 0.9},
                ],
            },
            "icd_codes": ["J18.9"],
            "retrieve_only": True,
        }
    )
    assert "query_too_vague" not in out["tags"]
    assert out["ranking_rating"] >= 3
