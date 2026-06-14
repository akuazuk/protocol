"""Audience hint and funnel rerank for adult vs pediatric protocols."""
from __future__ import annotations

from clinical_knowledge.methodist_search_ai_review import build_deterministic_search_ai_review
from rag_server import _demote_pediatric_for_adult_query, _rerank_protocols_symptom_only, doc_audience_hint


def test_doc_audience_hint_dет_nas_without_underscore():
    routing = {
        "pediatric_title_markers": ["дет нас", "дет_нас", "детск"],
        "adult_title_markers": ["взросл", "взр нас"],
    }
    hint = doc_audience_hint(
        "pulmon/epiglottitis.pdf",
        "Диагностика лечение эпиглоттита дет нас пост. МЗ 2023",
        routing,
    )
    assert hint == "pediatric"


def test_adult_query_demotes_pediatric_epiglottitis_with_r07():
    protos = [
        {
            "path": "a/epiglottitis_dets.pdf",
            "title": "Диагностика лечение эпиглоттита дет нас",
            "confidence_score": 0.98,
        },
        {
            "path": "b/pharyngitis_vzr.pdf",
            "title": "Диагностика лечение фарингита взр нас",
            "confidence_score": 0.75,
        },
    ]
    icd = {
        "explicit_icd_in_query": True,
        "detected": [{"code": "R07.0", "title_ru": "Боль в горле"}],
        "suggested": [],
        "codes_for_retrieval": ["R07.0"],
    }
    q = "болит горло\nКонтекст подбора: взрослое население\nМКБ-10: R07.0"
    out = _rerank_protocols_symptom_only(protos, q, icd)
    assert out[0]["path"].endswith("pharyngitis_vzr.pdf")


def test_demote_pediatric_for_adult_moves_children_to_end():
    protos = [
        {"path": "a/child.pdf", "title": "КП дет нас", "confidence_score": 0.99},
        {"path": "b/adult.pdf", "title": "КП взр нас", "confidence_score": 0.7},
    ]
    q = "кашель\nКонтекст подбора: взрослое население"
    out = _demote_pediatric_for_adult_query(protos, q)
    assert out[0]["path"].endswith("adult.pdf")


def test_methodist_deterministic_wrong_population_adult_top_pediatric():
    review = build_deterministic_search_ai_review(
        {
            "query": "болит горло\nКонтекст подбора: взрослое население",
            "llm_json": {
                "protocols": [
                    {
                        "path": "x/epiglottitis_dets.pdf",
                        "title": "эпиглоттит дет нас",
                    }
                ]
            },
            "funnel_context": {"population": "adult", "populationConfirmed": True},
            "retrieve_only": True,
            "icd_codes": ["R07.0"],
        }
    )
    assert "wrong_population" in review["tags"]
    assert review["top1_relevant"] is False
