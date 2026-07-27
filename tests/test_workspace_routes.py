from fastapi.testclient import TestClient

import rag_server


def test_workspace_routes_serve_index() -> None:
    client = TestClient(rag_server.app)
    for path in (
        "/doctor/search",
        "/doctor/review",
        "/doctor/recent",
        "/methodist/overview",
        "/methodist/cases",
        "/methodist/search-quality",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert "Найти протокол" in response.text
        assert response.headers["cache-control"] == "no-cache"


def test_workspace_html_uses_root_relative_catalog_asset() -> None:
    client = TestClient(rag_server.app)
    html = client.get("/doctor/search").text
    assert 'fetch("/protocols.json")' in html
    assert 'fetch("protocols.json")' not in html


def test_legacy_methodist_route_redirects_to_workspace() -> None:
    client = TestClient(rag_server.app)
    response = client.get("/methodist", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/methodist/overview"


def test_source_quality_requires_methodist_auth(monkeypatch) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "workspace-test-token")
    client = TestClient(rag_server.app)
    denied = client.get("/api/methodist/source-quality")
    assert denied.status_code == 403
    allowed = client.get(
        "/api/methodist/source-quality",
        headers={"X-Methodist-Token": "workspace-test-token"},
    )
    assert allowed.status_code == 200
    payload = allowed.json()
    assert {"summary", "queue_total", "top_queue"} <= payload.keys()
