"""SSE consult-review stream и inverted index."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_lex_inverted_index_built_after_load() -> None:
    import rag_server as rs

    rs._require_rag_loaded()
    assert isinstance(rs._lex_inverted_index, dict)
    if rs._chunks:
        assert len(rs._lex_inverted_index) > 0


def test_retrieve_with_inverted_index_matches_full_scan() -> None:
    import rag_server as rs

    rs._require_rag_loaded()
    q = "кашель бронхит J20"
    a = rs.retrieve(q, max_chunks=4, max_per_path=2)
    old = rs._lex_inverted_index
    try:
        rs._lex_inverted_index = {}
        b = rs.retrieve(q, max_chunks=4, max_per_path=2)
    finally:
        rs._lex_inverted_index = old
    assert [r.get("path") for r in a] == [r.get("path") for r in b]


def test_consult_stream_yields_progress_events(client, monkeypatch) -> None:
    import rag_server as rs

    rs._consult_review_cache.clear()

    def fake_iter(**kwargs):
        yield ("progress", {"stage": "test", "pct": 50, "label_ru": "тест", "partial": {}})
        yield (
            "done",
            {
                "ok": True,
                "review": {"overall_compliance_pct": 80, "criteria": [], "summary_ru": "ok"},
                "cached_result": False,
            },
        )

    monkeypatch.setattr(
        "consult_review_pipeline.iter_consult_review_pipeline",
        lambda **kw: fake_iter(**kw),
    )

    r = client.post(
        "/api/consult-review/stream",
        files=[("files", ("t.txt", b"test consult", "text/plain"))],
    )
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    body = r.text
    assert "progress" in body
    assert "done" in body
