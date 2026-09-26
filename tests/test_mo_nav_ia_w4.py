"""Меню без «Ещё» (редизайн v2, волна A): все пункты видны, Обзор = yesterday, алиасы URL включая /mis и /overview."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-ui.css").read_text(encoding="utf-8")
SERVER = (ROOT / "rag_server.py").read_text(encoding="utf-8")


def _nav() -> str:
    return HTML.split('id="app-nav"')[1].split("</ul>")[0]


def test_primary_nav_is_flat_without_more_menu() -> None:
    nav = _nav()
    assert "<ul" not in nav
    assert "nav-more" not in nav
    assert "Ещё" not in nav
    pages = []
    for line in nav.splitlines():
        if "nav-button" not in line or 'data-page="' not in line:
            continue
        assert " hidden" not in line.replace("aria-hidden", ""), line
        pages.append(line.split('data-page="', 1)[1].split('"', 1)[0])
    assert pages == [
        "yesterday", "documents", "mis", "doctors", "medications", "labs", "queue",
        "reports", "kp-sync", "rceth-sync", "settings",
    ]
    # «Период» слит с Обзором: кнопки нет, URL остаётся алиасом (см. test_url_aliases…).
    assert 'data-page="overview"' not in nav
    assert nav.count('class="nav-group-label"') == 2
    for label in ("Обзор", "Найти МО", "Поиск МИС", "Очередь", "Справка"):
        assert label in nav


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
    assert ".nav-group-label" in CSS


def test_url_aliases_keep_yesterday_and_queue() -> None:
    assert 'location.pathname.endsWith("/queue") ? "queue"' in APP
    assert 'location.pathname.endsWith("/overview") ? "overview"' in APP
    assert 'state.page === "queue" ? "/methodist/mo/queue"' in APP
    assert 'state.page === "overview" ? "/methodist/mo/overview"' in APP
    assert '@app.get("/methodist/mo/queue"' in SERVER
    assert 'location.pathname.endsWith("/mis") ? "mis"' in APP
    assert 'state.page === "mis" ? "/methodist/mo/mis"' in APP
    assert '@app.get("/methodist/mo/mis"' in SERVER


if __name__ == "__main__":
    test_primary_nav_is_flat_without_more_menu()
    test_overview_grain_and_titles()
    test_url_aliases_keep_yesterday_and_queue()
    print("ok")
