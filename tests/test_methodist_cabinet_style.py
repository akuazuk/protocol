"""Кабинет методиста: MO-style shell, full width, absolute links, no clutter tabs."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "frontend/web/doctor/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/methodist-cabinet.css").read_text(encoding="utf-8")
PATHS = (ROOT / "backend/frontend_paths.py").read_text(encoding="utf-8")


def test_methodist_cabinet_css_wired() -> None:
    assert 'href="/methodist-cabinet.css"' in INDEX
    assert '"methodist-cabinet.css"' in PATHS
    assert "body.methodist-mode .wrap" in CSS
    assert "min(1680px" in CSS
    assert "#app-chrome .app-chrome__journal" in CSS
    assert "#tab-methodist-b2c" in CSS
    assert "#section-footer .site-footer-nav" in CSS


def test_methodist_nav_compact_and_mo_link() -> None:
    assert 'id="tab-methodist-consult"' in INDEX
    assert 'id="methodist-nav-mo-analytics"' in INDEX
    assert 'href="/methodist/mo"' in INDEX
    assert "B2C · монетизация" not in INDEX
    assert 'href="/patient.html"' in INDEX
    assert 'href="/onco-risk.html"' in INDEX
    assert 'href="patient.html"' not in INDEX
    assert 'href="onco-risk.html"' not in INDEX
    assert 'href="docs/' not in INDEX


def test_methodist_cabinet_css_route() -> None:
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    response = client.get("/methodist-cabinet.css")
    assert response.status_code == 200
    assert "text/css" in (response.headers.get("content-type") or "")
    assert "body.methodist-mode" in response.text
    assert "no-cache" in (response.headers.get("cache-control") or "")
