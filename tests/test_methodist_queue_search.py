"""B4: methodist queue domain=search."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinical_knowledge.feedback_store import append_feedback_event
from clinical_knowledge.methodist_queue import build_methodist_queue, build_search_methodist_queue


@pytest.fixture
def feedback_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fb = tmp_path / "feedback"
    monkeypatch.setenv("ML_FEEDBACK_DIR", str(fb))
    return fb


@pytest.fixture
def methodist_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    pytest.importorskip("fastapi")
    import rag_server

    monkeypatch.setenv("METHODIST_TOKEN", "test-methodist-token")
    return TestClient(rag_server.app)


def test_build_kz_queue_has_domain(feedback_env):
    out = build_methodist_queue(limit=30)
    assert out.get("domain") == "kz"
    assert "priority" in out


def test_build_search_queue_empty(feedback_env):
    out = build_search_methodist_queue(limit=30)
    assert out["domain"] == "search"
    assert out["counts"]["search_retrieval_fix"] == 0


def test_build_search_queue_priority_on_bad_verdict(feedback_env):
    append_feedback_event(
        {
            "event_type": "search_review",
            "reviewer": "test",
            "query": "кашель температура",
            "ranking_verdict": "wrong",
            "retrieval_top_paths": ["minzdrav_protocols/x/a.pdf"],
            "tags": ["wrong_protocol"],
        }
    )
    out = build_methodist_queue(limit=30, domain="search")
    assert out["domain"] == "search"
    assert len(out["priority"]) >= 1
    row = out["priority"][0]
    assert row["verdict"] == "wrong"
    assert row.get("query_hash_short")
    assert "query" not in row


def test_api_methodist_queue_search_domain(methodist_client):
    r = methodist_client.get(
        "/api/methodist/queue",
        params={"domain": "search", "limit": 20},
        headers={"X-Methodist-Token": "test-methodist-token"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("domain") == "search"
