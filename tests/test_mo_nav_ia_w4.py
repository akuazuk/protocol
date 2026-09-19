"""W4: 5 рабочих пунктов + Ещё, Обзор = yesterday, алиасы URL."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-ui.css").read_text(encoding="utf-8")
SERVER = (ROOT / "rag_server.py").read_text(encoding="utf-8")


def _nav() -> str:
    return HTML.split('id="app-nav"')[1].split("</ul>")[0]


def test_primary_nav_is_five_plus_more() -> None:
    nav = _nav()
    assert "<ul" not in nav
    primary_pages = []
    in_more = False
    for line in nav.splitlines():
        if 'class="nav-more"' in line:
            in_more = True
        if "nav-button" not in line or "nav-settings" in line or 'data-page="' not in line:
            continue
        page = line.split('data-page="', 1)[1].split('"', 1)[0]
        if not in_more:
            primary_pages.append(page)
    assert primary_pages == ["yesterday", "documents", "doctors", "medications", "labs"]
    assert "Обзор" in nav
    assert "Найти МО" in nav
    assert "Ещё" in nav
    for extra in ("overview", "queue", "reports", "kp-sync", "rceth-sync"):
        assert f'data-page="{extra}"' in nav


def test_overview_grain_and_titles() -> None:
    assert 'id="title-yesterday">Обзор<' in HTML
    assert 'id="title-documents">Найти МО<' in HTML
    assert 'data-overview-grain="yesterday"' in HTML
    assert 'data-overview-grain="7d"' in HTML
    assert 'data-overview-grain="month"' in HTML
    assert 'yesterday: "Обзор"' in APP
    assert 'documents: "Найти МО"' in APP
    assert "function applyOverviewGrain" in APP
    assert "function syncOverviewGrain" in APP
    assert ".grain-strip" in CSS
    assert ".nav-more-menu" in CSS


def test_url_aliases_keep_yesterday_and_queue() -> None:
    assert 'location.pathname.endsWith("/queue") ? "queue"' in APP
    assert 'location.pathname.endsWith("/overview") ? "overview"' in APP
    assert 'state.page === "queue" ? "/methodist/mo/queue"' in APP
    assert 'state.page === "overview" ? "/methodist/mo/overview"' in APP
    assert '@app.get("/methodist/mo/queue"' in SERVER
    assert '@app.get("/methodist/mo/overview"' in SERVER


if __name__ == "__main__":
    test_primary_nav_is_five_plus_more()
    test_overview_grain_and_titles()
    test_url_aliases_keep_yesterday_and_queue()
    print("ok")
