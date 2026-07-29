import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return " ".join(parser.parts)


def test_mo_dashboard_has_complete_crm_navigation() -> None:
    for page in (
        "overview",
        "yesterday",
        "queue",
        "documents",
        "doctors",
        "specialties",
        "diagnoses",
        "safety",
        "data-quality",
        "reports",
        "settings",
    ):
        assert f'data-page="{page}"' in HTML


def test_mo_filters_are_multi_select_and_use_backend_contract() -> None:
    for key in ("months", "branches", "specialties", "doctors", "document_types", "statuses"):
        assert f'data-filter="{key}"' in HTML
    for api_key in ("periods", "filials", "specializations", "doctors", "document_kinds", "statuses"):
        assert f'"{api_key}"' in HTML
    assert "state.selected[key].join" in HTML
    assert 'id="case-search"' in HTML
    assert 'data-quick-period=' in HTML


def test_mo_dashboard_prefers_new_api_with_legacy_fallback() -> None:
    assert 'var API_ROOT = "/api/methodist/mo"' in HTML
    assert 'var LEGACY_ROOT = "/api/methodist/mis-kz-quality"' in HTML
    assert 'request("/overview"' in HTML
    assert 'request("/facets"' in HTML
    assert '"/freshness?"' in HTML


def test_mo_dashboard_accessibility_and_responsive_invariants() -> None:
    assert 'class="skip-link"' in HTML
    assert 'aria-live="polite"' in HTML
    assert 'role="dialog"' in HTML
    assert "@media (max-width: 720px)" in HTML
    assert "@media (prefers-reduced-motion: reduce)" in HTML


def test_user_facing_terminology_has_no_provider_or_internal_jargon() -> None:
    text = _visible_text(HTML)
    assert "МО Аналитика" in text
    assert "КЗ" not in text
    for forbidden in ("RAG", "LLM", "Gemini", "Render", "Cursor", "OpenAI", "Anthropic"):
        assert forbidden not in text


def test_mo_dashboard_javascript_has_valid_syntax() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", HTML, flags=re.DOTALL)
    result = subprocess.run(
        [node, "--check", "-"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cases_controls_are_wired_without_internal_status_prompt() -> None:
    assert '$("next-page").addEventListener("click"' in HTML
    assert '$("previous-page").addEventListener("click"' in HTML
    assert '$("columns-button").addEventListener("click"' in HTML
    assert 'id="bulk-status-value"' in HTML
    assert 'prompt("Статус:' not in HTML
    assert 'id="drawer-assignee"' in HTML
    assert 'id="drawer-due"' in HTML
    assert 'data-finding-code="' in HTML
    assert "История решений" in HTML
    assert '$("sort-by").addEventListener("change"' in HTML
    assert '$("sort-dir").addEventListener("change"' in HTML
