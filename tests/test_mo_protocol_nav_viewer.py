"""Навигатор КП: Study-reader, full-width МО, deep-link, оригинал PDF."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = (ROOT / "frontend/web/shared/proto-viewer.html").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-protocol-viewer.css").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
PATHS = (ROOT / "backend/frontend_paths.py").read_text(encoding="utf-8")


def test_viewer_uses_mo_tokens() -> None:
    assert 'href="/mo-tokens.css' in VIEWER
    assert 'href="/mo-protocol-viewer.css' in VIEWER
    assert "pv-mo" in VIEWER
    assert "Outfit" not in VIEWER
    assert "--accent" in CSS
    assert "min(1680px" in CSS


def test_viewer_reader_first_and_deep_links() -> None:
    assert "/api/protocol-reader" in VIEWER
    assert "fetchReader" in VIEWER
    assert "data-mode=\"study\"" in VIEWER
    assert "data-mode=\"visit\"" in VIEWER
    assert "/api/protocol-brief" in VIEWER
    assert 'qp("section")' in VIEWER
    assert 'qp("page")' in VIEWER
    assert "Оригинал · стр." in VIEWER
    assert "is-page-hit" in VIEWER
    assert "pv-para" in VIEWER
    assert "briefFromSourceDoc" not in VIEWER


def test_case_review_cta_labels() -> None:
    assert "Открыть протокол" in APP
    assert "Поиск в каталоге" in APP
    assert "function protocolViewerUrl" in APP
    assert "page=" in APP
    assert "from=mo" in APP
    assert "Открыть КП" not in APP


def test_frontend_path_maps_viewer_css() -> None:
    assert '"mo-protocol-viewer.css"' in PATHS


def test_viewer_css_has_stable_root_route() -> None:
    """Regression: HTML linked /mo-protocol-viewer.css but route was missing → 404 unstyled page."""
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    response = client.get("/mo-protocol-viewer.css")
    assert response.status_code == 200
    assert "text/css" in (response.headers.get("content-type") or "")
    assert ".pv-layout" in response.text
    assert "no-cache" in (response.headers.get("cache-control") or "")


def test_protocol_reader_api_route_exists() -> None:
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    # path required; empty corpus → available false, but route must respond
    response = client.get("/api/protocol-reader", params={"path": "missing/no-such.pdf"})
    assert response.status_code == 200
    body = response.json()
    assert "available" in body
    assert "paragraphs" in body
    assert body.get("build_version")
