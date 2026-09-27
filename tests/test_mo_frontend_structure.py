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
    # Канон меню: 5 рабочих пунктов + Ещё (Период/Очередь/Отчёты/КП/ЛС) + hidden settings.
    for page in (
        "overview",
        "yesterday",
        "queue",
        "documents",
        "doctors",
        "medications",
        "labs",
        "reports",
        "kp-sync",
        "rceth-sync",
        "settings",
    ):
        assert f'data-page="{page}"' in SOURCE
    for gone in ("specialties", "diagnoses", "safety", "doctor-cabinet"):
        assert f'id="page-{gone}"' not in HTML
    for label in (
        "Обзор",
        "Найти МО",
        "Очередь",
        "Врачи",
        "Лекарства",
        "Анализы",
        "Отчёты",
        "Протоколы МЗ",
        "Инструкции ЛС",
        "Справка",
    ):
        assert label in HTML
    assert "Ещё" not in HTML.split('id="app-nav"')[1].split("</ul>")[0]
    assert 'id="breadcrumbs"' in HTML
    assert 'id="doctor-zone-chart"' in HTML
    assert 'data-zone-preset="dx"' in HTML
    assert 'id="access-log-content"' in HTML  # secondary under Отчёты
    for chart_id in (
        "kp-sync-history-chart",
        "kp-sync-history-table",
        "kp-sync-month-chart",
        "kp-sync-year-chart",
        "kp-sync-slug-chart",
        "kp-sync-period-kpis",
        "kp-sync-period-table",
        "kp-sync-recent",
        "rceth-sync-kpis",
        "rceth-sync-live",
        "rceth-sync-live-title",
        "rceth-sync-live-text",
        "rceth-sync-history-table",
        "rceth-sync-freshness",
        "rceth-sync-notes",
    ):
        assert f'id="{chart_id}"' in HTML


def test_mo_filters_are_multi_select_and_use_backend_contract() -> None:
    for key in ("months", "branches", "specialties", "doctors", "document_types", "statuses"):
        assert f'data-filter="{key}"' in SOURCE
    for api_key in ("periods", "filials", "specializations", "doctors", "document_kinds", "crm_statuses"):
        assert f'"{api_key}"' in SOURCE
    # Волна J: фильтр статусов - это статус разбора методиста (CRM), не внутренний c.status.
    assert 'statuses: "crm_statuses"' in JS
    assert "Статус разбора методиста" in HTML
    assert 'statuses: values(rawFacets.crm_statuses' in JS
    # Legacy-значения шкалы из старых ссылок / представлений не уходят в crm_statuses.
    assert "CRM_STATUS_ORDER.indexOf(value) >= 0" in JS
    assert 'q.set(API_FILTER_KEYS[key] || key, chosen.join("|"))' in SOURCE
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
    assert "Протоколы МЗ" in JS
    assert "protocolViewerUrl" in JS
    assert "zone-card" in JS
    assert "Что не так" in JS
    assert "Разбор по критериям" in JS
    assert "zones-criteria-block" in JS
    assert "case-workspace-decision-scroll" in JS
    assert "case-workspace-grid--zones" in CSS or "case-workspace-grid--zones" in SOURCE
    assert "protocol-suggest-top" in JS
    assert 'id="drawer-pdf"' in HTML
    assert 'details open class="methodist-decision-panel methodist-decision-panel--dock"' in JS  # раскрыто по умолчанию (v2, волна A)
    assert "decision-dock-summary" in JS
    assert 'methodist-decision-panel--dock[open]' in CSS or "decision-dock-summary" in CSS
    assert "data-sort-key" in HTML
    assert 'id="drawer-prev"' in HTML
    assert "renderPatientHistory" in JS
    assert "renderLabBundle" in JS
    assert "renderLabReconcile" in JS
    assert "Лаборатория" in JS
    assert "B_lab_" in JS
    assert "Как история влияет на оценки" in JS
    assert "historyTierLabelRu" in JS
    assert "zoneFilter" in JS
    assert "ZONE_PRESETS" in JS


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
    assert "state.search = q.get(\"q\") || q.get(\"icd\") || \"\"" in JS
    assert "Pavel" not in SOURCE


def test_queue_has_no_only_critical_button_grade_chip_instead() -> None:
    # Волна J: «Только критические» заменён чипом оценки «Критично» в шапке.
    assert "queue-critical-only" not in SOURCE
    assert 'data-overall-grade="critical"' in HTML
    assert "Только критические" not in _visible_text(HTML)


def _table_headers(section_id: str, next_id: str) -> list[str]:
    chunk = HTML.split(f'id="{section_id}"')[1].split(f'id="{next_id}"')[0]
    return re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", chunk)


def test_case_lists_have_no_legacy_columns_and_no_p_levels() -> None:
    documents = _table_headers("page-documents", "page-doctors")
    queue = _table_headers("page-queue", "page-documents")
    day = _table_headers("page-yesterday", "page-queue")
    for legacy in ("Статус", "Итог", "Полнота проверки", "Надёжность", "Приоритет"):
        assert legacy not in documents, legacy
        assert legacy not in queue, legacy
        assert legacy not in day, legacy
    for wanted in ("МКБ", "История", "КП", "Оценка"):
        assert wanted in documents, wanted
    assert "Оценка" in queue and "Разбор" in queue
    assert day[0] == "Оценка"
    # Число ячеек строки = число колонок шапки; пустое состояние повторяет ту же ширину.
    assert "(queue ? 18 : 15)" in JS
    assert len(documents) == 15
    assert len(queue) == 18
    assert '"МКБ", "История", "КП", "Оценка"' in JS
    assert '"Ответственный", "Срок", "Разбор", "МО"' in JS
    # P0-P3 не попадают в строки очереди и таблицы дня: там оценка МО.
    for fn in ("function queueRow", "function documentRow", "function renderYesterdayActions"):
        start = JS.find(fn)
        assert start >= 0, fn
        body = JS[start : JS.find("\n    }\n", start)]
        assert "severityLabel(" not in body and "severityTone(" not in body, fn
        assert "overallGradeChip(" in body, fn


def test_no_hidden_hosts_and_no_dead_host_guard() -> None:
    # Пустые скрытые хосты (<div id="month-…" hidden></div>) - мёртвая разметка; баннеры
    # состояния (month-reconciliation, partial-banner) скрыты по смыслу и остаются.
    assert not re.search(r'id="(?:month|yesterday)-[a-z-]+"\s+hidden(?:\s+aria-hidden="true")?>\s*(?:<option[^<]*</option>\s*)?</(?:div|select)>', HTML)
    assert not re.search(r'class="nav-button"[^>]*\shidden', HTML)
    assert "hostActive" not in JS


def test_column_presets_are_primary_and_not_collapsible() -> None:
    start = JS.find("function renderColumnsManager")
    body = JS[start : start + 3000]
    assert 'presetButton("work"' in body and 'presetButton("review"' in body
    assert "<details" not in body
    assert "columnPresetActive(key)" in body
    assert "aria-pressed" in body


def test_server_tables_map_chrome_to_query_params() -> None:
    start = JS.find("function attachTableChrome")
    body = JS[start : start + 6000]
    assert "data-table-server-search" in body
    assert 'data-server-grade="' in body
    assert "setOverallGrade(value, { force: true })" in body
    assert 'var BAD_GRADES = "critical|important|poor";' in JS
    assert "state.search = next;" in body


def test_facets_are_fetched_for_non_overview_pages_and_on_panel_open() -> None:
    """/cases не отдаёт facets: при deep-link на «Найти МО»/«Очередь» меню фильтров
    должны заполняться отдельным запросом /facets (и при открытии панели)."""
    assert "async function ensureFacets(force)" in JS
    assert 'request("/facets?" + key, "/cases?" + key)' in JS
    load_page = JS[JS.find("async function loadPage(page)") :]
    load_page = load_page[: load_page.find("function savedViews")]
    assert 'if (page !== "overview") ensureFacets().catch(function () {});' in load_page
    draft = JS[JS.find("function beginFilterDraft()") :][:400]
    assert "ensureFacets().catch(function () {});" in draft
    # ключ кэша не зависит от пагинации/сортировки
    key_fn = JS[JS.find("function facetsQueryKey()") :][:300]
    for param in ("page", "page_size", "sort_by", "sort_dir"):
        assert f'"{param}"' in key_fn
    # поздний ответ /facets перерисовывает меню - набранный поиск по фильтру сохраняется,
    # а отметки берутся из filterDraft/selected (publishFacet пишет туда сразу)
    render = JS[JS.find("function renderFilter(details)") :][:2600]
    assert "var previousTerm = previousSearch ? previousSearch.value : \"\";" in render
    assert "if (previousTerm) {" in render
    assert "state.filterDraft.selected" in render
    # поздний ответ /facets перерисовывает меню - набранный поиск по фильтру сохраняется,
    # а отметки берутся из filterDraft/selected (publishFacet пишет туда сразу)
    render = JS[JS.find("function renderFilter(details)") :][:2600]
    assert "var previousTerm = previousSearch ? previousSearch.value : \"\";" in render
    assert "if (previousTerm) {" in render
    assert "state.filterDraft.selected" in render


def test_facet_checkbox_publishes_without_waiting_outer_apply() -> None:
    assert "function publishFacet(next, closeMenu)" in JS
    assert "publishFacet(draft, false)" in JS


def test_cases_table_keeps_rows_while_reloading() -> None:
    assert "function setCasesLoading(queue, on)" in JS
    assert "if (body.querySelector(\"tr[data-case]\")) return;" in JS
    assert "setCasesLoading(queue, true)" in JS
    assert "beginPageRequestScope()" in JS
    assert "is-skeleton-row" in JS
    assert "table-wrap.is-loading" in CSS
    assert "drawer-body.is-loading" in CSS
    assert "keepBody" in JS
    assert "function paintCaseChrome(item)" in JS
    assert "case-inspector-pending" in JS
    assert "caseDetailLoading" in JS
    idx = JS.find("label: \"Показать критические случаи\"")
    assert idx >= 0
    chunk = JS[idx : idx + 280]
    assert 'applyQueueBand("critical"' in chunk
    assert 'statuses = ["Критично"]' not in chunk


def test_month_reconciliation_hides_zero_delta_banner() -> None:
    assert "Number(reconciliation.source_delta || 0) === 0" in JS
    assert "Number(reconciliation.evaluated_delta || 0) === 0" in JS
    assert '$("title-yesterday").textContent = "Обзор"' in JS


def test_find_bar_exposes_grade_strip_and_icd_query() -> None:
    assert 'id="grade-strip"' in HTML
    assert 'data-overall-grade="good"' in HTML
    assert 'id="documents-queue-only"' in HTML
    assert "function looksLikeIcd(text)" in JS
    assert 'q.set("icd", searchRaw.toUpperCase())' in JS
    assert "function setOverallGrade(grade, opts)" in JS
    assert "minmax(360px, 1fr)" in CSS
    assert "state.queueOnly" in JS
    assert "function applyQueueBand(band, opts)" in JS
    assert 'q.set("queue_band", state.queueBand)' in JS


def test_mo_api_request_does_not_fetch_undefined_legacy() -> None:
    api = (SHARED / "mo-api.js").read_text(encoding="utf-8")
    assert 'legacy == null || legacy === ""' in api
    assert "mis-kz-qualityundefined" not in api


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
    assert "/review-pack" in JS
    assert "review-pack" in JS
    assert '$("sort-by").addEventListener("change"' in JS
    assert '$("sort-dir").addEventListener("change"' in JS


def test_case_drawer_renders_source_mo_and_never_turns_missing_scores_into_zero() -> None:
    assert "function renderClinicalDocument" in JS
    assert "function renderMonthReg55Section" in JS
    assert "function reg55BandPill" in JS
    assert "Соответствие №55" in JS
    assert 'id="month-rubric-mz"' in HTML
    assert 'id="month-reg55"' in HTML
    assert "/reg55-section-summary?" in JS
    assert "state.rubricCriterion" in JS
    assert 'data-rubric-criterion="' in JS
    assert 'data-reg55-band="' in JS
    assert "reg55_point" in JS
    assert "reg55_band" in JS
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


def test_overview_f1_dashboards_are_present_and_have_no_collapsibles() -> None:
    """Волна F1: Обзор O1-O8 - 6 диаграмм, компактная таблица дня и полнота без раскрывашек."""
    page = re.search(r'<section class="page" id="page-yesterday".*?</section>', HTML, re.S)
    assert page is not None
    overview = page.group(0)
    assert "<details" not in overview, "на Обзоре не должно быть скрытых блоков"
    for host in (
        "yesterday-grade-band",
        "yesterday-score-rings",
        "yesterday-score-dynamics",
        "yesterday-kp-funnel",
        "yesterday-heatmap",
        "yesterday-findings-top",
        "yesterday-action-rows",
        "yesterday-completeness",
    ):
        assert f'id="{host}"' in overview, host
    headers = re.findall(r"<th>([^<]+)</th>", re.search(r'<table class="table-dense">.*?</thead>', overview, re.S).group(0))
    assert headers == ["Оценка", "Визит", "Дата", "Врач / специальность", "Филиал", "Диагноз", "Причина", "МО"]
    app = (SHARED / "mo-app.js").read_text(encoding="utf-8")
    assert '"/overview-dashboard?"' in app
    for renderer in ("renderGradeBand", "renderKpFunnel", "renderSpecialtyHeatmap", "renderFindingsTop"):
        assert f"function {renderer}(dash)" in app, renderer
        assert f"{renderer}(dash);" in app, renderer
    assert "trends_compare" in app, "сравнение с прошлым периодом рисуется пунктиром"
    assert 'id: "chrome-yesterday-action-rows", dense: true' in app
    assert "color-mix(" not in re.search(r"function mixHex.*?function isoDate", app, re.S).group(0)
    assert ".table-dense th, .table-dense td { padding: 4px 8px" in CSS
