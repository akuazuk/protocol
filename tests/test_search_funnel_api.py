"""Tests for POST /api/search/funnel (C5)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    pytest.importorskip("fastapi")
    import rag_server

    return TestClient(rag_server.app)


def test_search_funnel_step0(client: TestClient) -> None:
    r = client.post(
        "/api/search/funnel",
        json={"query": "кашель и температура", "step": 0, "context": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 0
    assert body.get("valid") is True
    assert body.get("session_id")


def test_search_funnel_step1_population_choices(client: TestClient) -> None:
    r = client.post(
        "/api/search/funnel",
        json={"query": "кашель и температура 39", "step": 1, "context": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 1
    assert body.get("choices")


def test_search_funnel_step1_auto_skip_with_context(client: TestClient) -> None:
    r = client.post(
        "/api/search/funnel",
        json={
            "query": "кашель",
            "step": 1,
            "context": {"population": "adult"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("auto_skip") is True
