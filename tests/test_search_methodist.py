"""Tests for search methodist P0 APIs."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    pytest.importorskip("fastapi")
    import rag_server

    return TestClient(rag_server.app)


@pytest.fixture
def methodist_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("METHODIST_TOKEN", "test-methodist-token")
    return client


def test_protocol_summary_nav_ok(client: TestClient) -> None:
    r = client.get(
        "/api/protocol-summary-nav",
        params={"path": "minzdrav_protocols/gastroenterologiya/nonexistent.pdf"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert body["path"]


def test_protocol_summary_excerpt_ok(client: TestClient) -> None:
    r = client.get(
        "/api/protocol-summary-excerpt",
        params={
            "path": "minzdrav_protocols/gastroenterologiya/nonexistent.pdf",
            "condition_id": "c1",
            "section_id": "criteria",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("llm_used") is False
    assert body["available"] is False


def test_methodist_protocol_search_requires_auth(client: TestClient) -> None:
    r = client.get("/api/methodist/protocol-search", params={"q": "гастро"})
    assert r.status_code in (403, 503)


def test_methodist_protocol_search_ok(methodist_client: TestClient) -> None:
    r = methodist_client.get(
        "/api/methodist/protocol-search",
        params={"q": "протокол"},
        headers={"X-Methodist-Token": "test-methodist-token"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "протокол"
    assert "items" in body
    assert isinstance(body["items"], list)
