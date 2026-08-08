import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
SHARED = ROOT / "frontend" / "web" / "shared"
CSS = "\n".join((SHARED / name).read_text(encoding="utf-8") for name in ("mo-tokens.css", "mo-ui.css"))
APP_JS_FILES = tuple(SHARED / name for name in ("mo-api.js", "mo-charts.js", "mo-app.js"))
JS_FILES = APP_JS_FILES + (SHARED / "vendor" / "echarts.min.js",)
JS = "\n".join(path.read_text(encoding="utf-8") for path in APP_JS_FILES)
SOURCE = "\n".join((HTML, CSS, JS))


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
        assert f'data-page="{page}"' in SOURCE
    for section in ("Мониторинг", "Работа", "Аналитика", "Управление"):
        assert f'<li class="nav-section-label">{section}</li>' in HTML
    assert 'id="breadcrumbs"' in HTML


def test_mo_filters_are_multi_select_and_use_backend_contract() -> None:
    for key in ("months", "branches", "specialties", "doctors", "document_types", "statuses"):
        assert f'data-filter="{key}"' in SOURCE
    for api_key in ("periods", "filials", "specializations", "doctors", "document_kinds", "statuses"):
        assert f'"{api_key}"' in SOURCE
    assert 'state.selected[key].join("|")' in SOURCE
    assert 'id="case-search"' in SOURCE
    assert 'data-quick-period=' in SOURCE
    assert 'id="score-eligible-only"' in SOURCE
    assert "score_eligible_only" in SOURCE
    assert 'document_types: ["clinical_visit"]' in SOURCE
    assert 'id="score-eligible-only" checked disabled' in HTML
    assert 'q.set("score_eligible_only", "1")' in SOURCE
    assert "URL score_eligible_only=0" in SOURCE or "score_eligible_only=0" in SOURCE


def test_case_workspace_has_dual_scroll_and_large_summary() -> None:
    assert "case-workspace-clinical" in JS
    assert "case-workspace-decision" in JS
    assert 'id="drawer-summary"' in JS
    assert "maxlength=\"12000\"" in JS or "maxlength=\\\"12000\\\"" in JS or "maxlength=\"12000\"" in SOURCE
    assert "drawer-score-c" not in JS
    assert "Полнота %" not in JS
    assert "protocol-suggest" in JS
    assert "Подбор по установленному диагнозу" in JS
    assert "data-sort-key" in HTML
    assert 'id="drawer-prev"' in HTML
    assert "renderPatientHistory" in JS
    assert "Как история влияет на оценки" in JS
    assert "historyTierLabelRu" in JS


def test_mo_search_and_filters_have_explicit_apply_actions() -> None:
    assert 'id="case-search-form"' in HTML
    assert 'id="case-search-submit"' in HTML
    assert 'id="case-search-clear"' in HTML
    assert 'id="filters-panel"' in HTML
    assert 'id="views-panel"' in HTML
    assert 'data-filter-apply' in JS
    assert 'data-filter-clear' in JS
    assert '$("case-search-form").addEventListener("submit"' in JS
    assert '$("case-search").addEventListener("change"' not in JS
    assert "state.search = q.get(\"q\") || \"\"" in JS
    assert "Pavel" not in SOURCE


def test_mo_dashboard_prefers_new_api_with_legacy_fallback() -> None:
    assert 'var API_ROOT = "/api/methodist/mo"' in SOURCE
    assert 'var LEGACY_ROOT = "/api/methodist/mis-kz-quality"' in SOURCE
    assert 'request("/overview"' in SOURCE
    assert 'request("/facets"' in SOURCE
    assert '"/freshness?"' in SOURCE


def test_mo_dashboard_accessibility_and_responsive_invariants() -> None:
    assert 'class="skip-link"' in HTML
    assert 'aria-live="polite"' in HTML
    assert 'role="dialog"' in HTML
    assert "@media (max-width: 720px)" in CSS
    assert "@media (prefers-reduced-motion: reduce)" in CSS
    assert '<caption class="sr-only">Очередь случаев для разбора методистом</caption>' in HTML
    assert '<caption class="sr-only">Все медицинские документы выбранного среза</caption>' in HTML
    assert 'scope="col"' in HTML


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
    inline_scripts = re.findall(r"<script[^>]*>(.*?)</script>", HTML, flags=re.DOTALL)
    assert not any(script.strip() for script in inline_scripts)
    for path in JS_FILES:
        result = subprocess.run(
            [node, "--check", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_cases_controls_are_wired_without_internal_status_prompt() -> None:
    assert '$("next-page").addEventListener("click"' in JS
    assert '$("previous-page").addEventListener("click"' in JS
    assert '$("columns-button").addEventListener("click"' in JS
    assert 'id="bulk-status-value"' in HTML
    assert 'prompt("Статус:' not in SOURCE
    assert 'id="drawer-assignee"' in JS
    assert 'id="drawer-due"' in JS
    assert 'data-finding-code="' in JS
    assert "История разборов" in JS
    assert "История CRM" in JS
    assert "Сохранить пакет разбора" in JS
    assert "/review-pack" in JS
    assert '$("sort-by").addEventListener("change"' in JS
    assert '$("sort-dir").addEventListener("change"' in JS


def test_case_drawer_renders_source_mo_and_never_turns_missing_scores_into_zero() -> None:
    assert "function renderClinicalDocument" in JS
    assert "function renderRubricMz" in JS
    assert "function renderMonthRubricMz" in JS
    assert "Рубрика МЗ" in JS
    assert "Как оценивать" in JS
    assert 'id="month-rubric-mz"' in HTML
    assert '"/rubric-summary?"' in JS
    assert "state.rubricCriterion" in JS
    assert 'data-rubric-criterion="' in JS
    assert 'kpi("Рубрика МЗ"' in JS
    for field in ("complaints", "anamnesis_doctor", "objective_status", "clinical_diagnosis"):
        assert f'["{field}"' in JS
    assert 'available ? Math.round(n) + "%" : "Нет данных"' in JS
    assert 'unscored:"Не оценено"' in JS
    assert 'documentData.source_format === "secure_csv"' in JS
    assert "Клинический текст недоступен" in JS


def test_health_and_capabilities_are_rendered_without_guessing_features() -> None:
    assert 'request("/capabilities", "/meta")' in JS
    assert 'request("/health", "/freshness")' in JS
    assert 'id="health-components"' in HTML
    assert "case_document_source" in JS


def test_programmatic_main_focus_does_not_draw_workspace_frame() -> None:
    assert ".content:focus { outline: none; }" in SOURCE
