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
        "/methodist/mis-kz",
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


def test_methodist_mo_dashboard_has_canonical_routes() -> None:
    client = TestClient(rag_server.app)
    for path in ("/methodist/mo", "/methodist/mo/yesterday", "/methodist/mo/cases"):
        response = client.get(path)
        assert response.status_code == 200
        assert "МО Аналитика" in response.text
        assert response.headers["cache-control"] == "no-cache, must-revalidate"
        assert 'href="/methodist/mis-kz"' in response.text
        assert '"/api/methodist/mo"' in response.text


def test_legacy_methodist_quality_routes_redirect_to_mo() -> None:
    client = TestClient(rag_server.app)
    for path in ("/methodist/mis-kz-quality", "/methodist/mis-kz-quality.html"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/methodist/mo"


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


def test_mis_kz_summary_api_reads_current_project_export(monkeypatch) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "mis-kz-test-token")
    client = TestClient(rag_server.app)
    response = client.get(
        "/api/methodist/mis-kz-quality?month=2026-07&compare_month=2026-01",
        headers={"X-Methodist-Token": "mis-kz-test-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["available"] is True
    assert payload["month"] == "2026-07"
    assert payload["n_cases"] > 0
    assert payload["doctors_n"] > 0
