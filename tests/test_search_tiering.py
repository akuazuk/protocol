"""S0/S1/S2 tiering, /api/search/run, health search metrics."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clinical_knowledge.search_tiering import (
    apply_search_tier_flags,
    build_search_path_allowlist,
    query_has_icd_code,
    resolve_search_tier,
    search_require_allowlist,
)
from clinical_knowledge.search_run import run_search_request


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_resolve_search_tier_defaults_and_normalizes():
    assert resolve_search_tier(None) == "S1"
    assert resolve_search_tier("s2") == "S2"
    assert resolve_search_tier("bogus") == "S1"


def test_query_has_icd_code():
    assert query_has_icd_code("J06.9 кашель")
    assert not query_has_icd_code("кашель без кода")


def test_s0_requires_icd():
    flags = apply_search_tier_flags("S0", explicit_icd_codes=[], query="кашель")
    assert flags.get("error")
    flags_ok = apply_search_tier_flags("S0", explicit_icd_codes=["J06.9"], query="кашель")
    assert flags_ok.get("tier") == "S0"
    assert flags_ok.get("require_icd_fast") is True


def test_s2_enables_full_assist_flags():
    flags = apply_search_tier_flags("S2", explicit_icd_codes=[], query="кашель")
    assert flags.get("retrieve_only") is False
    assert flags.get("tier") == "S2"


def test_build_search_path_allowlist_merges_sources():
    merged = build_search_path_allowlist(
        path_allowlist=["minzdrav_protocols/a/x.pdf"],
        icd_lookup_allowlist=["minzdrav_protocols/b/y.pdf"],
        icd_codes=None,
        path_boost=["minzdrav_protocols/a/x.pdf"],
        search_ctx={"path_boost": ["minzdrav_protocols/c/z.pdf"]},
    )
    assert merged == [
        "minzdrav_protocols/a/x.pdf",
        "minzdrav_protocols/b/y.pdf",
        "minzdrav_protocols/c/z.pdf",
    ]


def test_run_search_request_routes_funnel_when_step_ge_zero():
    out = run_search_request(
        query="кашель",
        tier="S1",
        step=2,
        context={"population": "adult"},
        category_slugs=[],
        icd_codes=[],
        funnel_population=None,
        session_id=None,
        assist_fn=lambda **kw: {"assist": True},
        funnel_fn=lambda **kwargs: {"step": kwargs["step"], "ok": True},
        protocols_by_icd_fn=lambda **kw: {"icd": True},
    )
    assert out.get("step") == 2
    assert out.get("search_tier") == "S1"


def test_run_search_request_s0_uses_protocols_by_icd():
    calls: list[str] = []

    def _by_icd(**kwargs):
        calls.append("icd")
        return {"llm_json": {"protocols": []}}

    out = run_search_request(
        query="J06.9",
        tier="S0",
        step=-1,
        context={},
        category_slugs=[],
        icd_codes=["J06.9"],
        funnel_population=None,
        session_id=None,
        assist_fn=lambda **kw: pytest.fail("assist must not run"),
        funnel_fn=lambda **kw: pytest.fail("funnel must not run"),
        protocols_by_icd_fn=_by_icd,
    )
    assert calls == ["icd"]
    assert out.get("search_tier") == "S0"
    assert out.get("ok") is True


def test_api_search_run_s0(client, monkeypatch):
    monkeypatch.setattr(
        "rag_server.api_search_protocols_by_icd",
        lambda body: {
            "llm_json": {"protocols": [{"path": "minzdrav_protocols/x/a.pdf"}]},
            "icd_fast_lookup": True,
        },
    )
    r = client.post(
        "/api/search/run",
        json={"query": "J06.9 ОРВИ", "tier": "S0", "icd_codes": ["J06.9"], "step": -1},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("search_tier") == "S0"
    assert data.get("ok") is True


def test_assist_search_tier_s2_not_retrieve_only(client, monkeypatch):
    import rag_server as rs

    captured: dict = {}

    def _fake_impl(body):
        captured["retrieve_only"] = body.retrieve_only
        captured["search_tier"] = body.search_tier
        return {
            "query": body.query,
            "retrieve_only": False,
            "search_tier": "S2",
            "llm_json": {"protocols": []},
        }

    monkeypatch.setattr(rs, "_api_assist_impl", _fake_impl)
    r = client.post(
        "/api/assist",
        json={"query": "кашель 3 дня", "search_tier": "S2"},
    )
    assert r.status_code == 200, r.text
    assert captured.get("search_tier") == "S2"


def test_health_includes_search_metrics(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "search_concurrency" in data
    assert "search_last" in data


def test_search_require_allowlist_on_render(monkeypatch):
    monkeypatch.delenv("RAG_SEARCH_REQUIRE_ALLOWLIST", raising=False)
    monkeypatch.setenv("RENDER", "1")
    monkeypatch.setenv("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", "1")
    assert search_require_allowlist() is True
