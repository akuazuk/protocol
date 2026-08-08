import re
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "frontend" / "web"
SHARED = WEB / "shared"
HTML = (WEB / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
TOKENS = (SHARED / "mo-tokens.css").read_text(encoding="utf-8")
UI = (SHARED / "mo-ui.css").read_text(encoding="utf-8")
API = (SHARED / "mo-api.js").read_text(encoding="utf-8")
CHARTS = (SHARED / "mo-charts.js").read_text(encoding="utf-8")
APP = (SHARED / "mo-app.js").read_text(encoding="utf-8")


def test_markup_shell_is_small_and_has_no_inline_executable_assets() -> None:
    assert len(HTML.splitlines()) < 380
    for asset in ("mo-tokens.css", "mo-ui.css", "mo-api.js", "mo-charts.js", "mo-app.js"):
        assert f'/{asset}' in HTML
    assert "<style" not in HTML
    assert not re.search(r"<script(?![^>]+src=)", HTML)
    assert not re.search(r"\son[a-z]+=", HTML)


def test_ordered_namespace_and_legacy_api_fallback_are_explicit() -> None:
    assert "window.MO = window.MO || {}" in API
    assert "window.MO = window.MO || {}" in CHARTS
    assert "window.MO = window.MO || {}" in APP
    assert HTML.index("/mo-api.js") < HTML.index("/mo-charts.js") < HTML.index("/mo-app.js")
    assert '"/api/methodist/mo"' in API
    assert '"/api/methodist/mis-kz-quality"' in API
    assert "response.status === 404" in API


def test_semantic_tokens_dark_theme_density_and_control_size() -> None:
    for token in ("--sev-p0", "--sev-p1", "--sev-p2", "--good", "--warn", "--bad", "--neutral"):
        assert token in TOKENS
    assert ':root[data-theme="dark"]' in TOKENS
    assert ':root[data-density="compact"]' in TOKENS
    assert "prefers-color-scheme: dark" in TOKENS
    assert "--control-height: 44px" in TOKENS
    assert "localStorage.setItem(THEME_KEY" in APP
    assert "localStorage.setItem(DENSITY_KEY" in APP


def test_shared_shell_accessibility_and_keyboard_features() -> None:
    for marker in (
        'class="sidebar"',
        'class="context-bar"',
        'class="grid"',
        'id="case-drawer"',
        'id="toast-region"',
        'id="command-palette"',
        'id="search-suggestions"',
    ):
        assert marker in HTML
    assert 'role="dialog"' in HTML
    assert 'aria-live="polite"' in HTML
    assert "event.metaKey || event.ctrlKey" in APP
    assert 'event.key.toLowerCase() === "k"' in APP
    assert 'event.key === "Tab"' in APP
    assert "state.trigger.focus()" in APP
    assert "state.commandTrigger" in APP
    assert "window.setTimeout(function () { renderSearchSuggestions(value); }, 250)" in APP
    assert "@media (prefers-reduced-motion: reduce)" in UI


def test_echarts_is_self_hosted_wrapped_and_has_fallback() -> None:
    runtime = SHARED / "vendor" / "echarts.min.js"
    license_file = SHARED / "vendor" / "ECHARTS-LICENSE.txt"
    assert runtime.stat().st_size > 100_000
    assert "Apache License" in license_file.read_text(encoding="utf-8")
    assert 'src="/vendor/echarts.min.js"' in HTML
    assert "MO.moChart = moChart" in CHARTS
    assert "aria" in CHARTS and "enabled: true" in CHARTS
    assert "ResizeObserver" in CHARTS and 'window.addEventListener("resize"' in CHARTS
    assert "prefers-reduced-motion: reduce" in CHARTS
    assert "exportChartPng" in CHARTS
    assert "typeof config.fallback" in CHARTS
    assert "Array.isArray(axis)" in CHARTS
    assert "themeAxis" in CHARTS
    assert "renderTrendChart" in APP and "MO.moChart(element" in APP


def test_reports_page_has_interactive_cards_and_kpi_strip() -> None:
    assert 'id="report-kpis"' in HTML
    assert 'class="report-grid"' in HTML
    assert 'data-report-date="' in APP
    assert "filtersToSearchParams" in APP
    assert 'daily-report?date=' in APP


def test_page_references_no_external_url_or_cdn_and_has_no_long_dash() -> None:
    assert not re.search(r'(?:src|href)=["\']https?://', HTML)
    assert "cdn." not in HTML.lower()
    for source in (HTML, TOKENS, UI, API, CHARTS, APP):
        assert "\u2013" not in source
        assert "\u2014" not in source


def test_static_assets_have_stable_routes_content_types_and_no_cache() -> None:
    client = TestClient(rag_server.app)
    expected = {
        "/mo-tokens.css": "text/css",
        "/mo-ui.css": "text/css",
        "/mo-api.js": "application/javascript",
        "/mo-charts.js": "application/javascript",
        "/mo-app.js": "application/javascript",
        "/vendor/echarts.min.js": "application/javascript",
        "/vendor/ECHARTS-LICENSE.txt": "text/plain",
        "/protocol-chrome-tabs.css": "text/css",
        "/search-flow.css": "text/css",
        "/search-flow.js": "application/javascript",
        "/ux-redesign.css": "text/css",
        "/protocol-logo.svg": "image/svg+xml",
        "/protocol-logo-mini.svg": "image/svg+xml",
        "/protocol-logo-wordmark.svg": "image/svg+xml",
    }
    for route, content_type in expected.items():
        response = client.get(route)
        assert response.status_code == 200, route
        assert response.headers["content-type"].startswith(content_type)
        assert response.headers["cache-control"] == "no-cache, must-revalidate"
