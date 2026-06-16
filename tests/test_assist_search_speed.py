"""S1–S5: быстрый поиск — ICD auto fast path, retrieve_only без Gemini, search_timing."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_icd_codes_for_fast_lookup_from_query_text():
    import rag_server as rs
    from icd_mkb import analyze_query_for_icd

    icd = analyze_query_for_icd("J06.9 ОРВИ кашель", "J06.9 ОРВИ кашель")
    codes = rs._icd_codes_for_fast_lookup(body_codes=[], icd_analysis=icd)
    assert "J06.9" in codes


def test_assist_auto_icd_fast_path_from_query_text(client, monkeypatch):
    import rag_server as rs

    monkeypatch.setenv("RAG_ICD_FAST_AUTO", "1")
    calls: list[str] = []

    def _fake_fast(**kwargs):
        calls.append("fast")
        return {
            "query": kwargs["query"],
            "retrieve_only": True,
            "assist_lite": True,
            "icd_fast_lookup": True,
            "lookup_ms": 12,
            "llm_json": {"protocols": [{"path": "minzdrav_protocols/x/a.pdf", "title": "A"}]},
            "finish_reason": "ICD_LOOKUP",
        }

    monkeypatch.setattr(rs, "_try_icd_fast_assist", _fake_fast)
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: pytest.fail("retrieve must not run"))

    icd_analysis = {
        "codes_for_retrieval": ["J06.9"],
        "detected": [{"code": "J06.9", "title_ru": "ОРВИ"}],
        "suggested": [],
    }
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda query, model, **kw: (icd_analysis, query, query, None, None),
    )

    r = client.post(
        "/api/assist",
        json={"query": "J06.9 ОРВИ кашель", "retrieve_only": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("icd_fast_lookup") is True
    assert data.get("search_timing", {}).get("path") == "icd_fast_lookup"
    assert calls == ["fast"]


def test_retrieve_only_skips_specialty_gemini(client, monkeypatch):
    import rag_server as rs

    monkeypatch.setenv("RAG_SEARCH_SKIP_LLM_ON_RETRIEVE_ONLY", "1")
    icd_analysis = {"codes_for_retrieval": [], "detected": [], "suggested": []}
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda query, model, **kw: (icd_analysis, query, query, None, None),
    )
    monkeypatch.setattr(rs, "_try_icd_fast_assist", lambda **kw: None)

    def _no_specialty(q, model):
        raise AssertionError("infer_specialties_gemini must be skipped on retrieve_only")

    monkeypatch.setattr(rs, "infer_specialties_gemini", _no_specialty)
    fake_rows = [
        {
            "path": "minzdrav_protocols/pulmonologiya/a.pdf",
            "kind": "treatment",
            "excerpt": "x",
            "score": 1.0,
            "lexical_score": 1.0,
            "routing_multiplier": 1.0,
        }
    ]
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: list(fake_rows))
    monkeypatch.setattr(
        rs, "filter_retrieval_by_audience", lambda rows, q, routing: (rows, None, False)
    )

    r = client.post("/api/assist", json={"query": "кашель температура", "retrieve_only": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("retrieve_only") is True
    timing = data.get("search_timing") or {}
    assert timing.get("path") == "retrieve_only"
    assert timing.get("total_ms", 0) >= 0


def test_protocols_by_icd_includes_search_timing(client):
    r = client.post(
        "/api/search/protocols-by-icd",
        json={"query": "ОРВИ", "icd_codes": ["J06.9"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("icd_fast_lookup") is True
    timing = data.get("search_timing") or {}
    assert timing.get("path") == "icd_fast_lookup"
    assert timing.get("lookup_ms", 9999) < 500
