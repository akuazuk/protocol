"""Тесты batch probe поиска протоколов."""
from __future__ import annotations

from clinical_knowledge.methodist_search_probe_runner import (
    build_probe_query,
    summarize_probe_reports,
)


def test_build_probe_query_population_and_icd():
    q = build_probe_query(
        {
            "query": "кашель",
            "population": "pediatric",
            "icd_codes": ["J06.9"],
        }
    )
    assert "кашель" in q
    assert "детское население" in q
    assert "J06.9" in q


def test_summarize_probe_reports_hit_rates():
    reports = [
        {"expected_hit1": True, "ai_rating": 4, "ai_verdict": "mostly_correct"},
        {"expected_hit1": False, "ai_rating": 3, "ai_verdict": "partially_wrong"},
    ]
    s = summarize_probe_reports(reports)
    assert s["n_ok"] == 2
    assert s["expected_hit1_pct"] == 0.5
    assert s["avg_ai_rating"] == 3.5
