"""W1: calendar in header, dates always apply, warehouse tables without col-filters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_header_exposes_calendar_without_apply_buttons() -> None:
    assert 'id="period-strip"' in HTML
    assert 'id="date-from-wrap"' in HTML
    assert 'data-quick-period="custom"' in HTML
    assert 'id="date-from-wrap" hidden' in HTML
    assert 'id="filters-apply"' not in HTML
    assert 'id="filters-cancel"' not in HTML
    assert 'id="reset-filters"' in HTML
    assert "Сбросить всё" in HTML
    assert 'placeholder="Врач, диагноз, МКБ, visit_id"' in HTML


def test_query_always_sends_date_window() -> None:
    assert "function applyPeriodPreset(period, opts)" in APP
    assert 'q.set("date_from", state.dateFrom)' in APP
    assert "if (!st.serverSort && !st.dense)" in APP
    # Волна J: chrome серверных таблиц - это параметры запроса, не фильтр строк экрана.
    assert "Среди строк на экране" not in APP
    assert "Поиск по всей выборке" in APP
    assert "Диагноз не найден" in APP
    assert "chip-reset-all" not in APP
    assert "function syncCustomDates" in APP
    assert "function ensureTableLoadBar" in APP
    assert "is-refreshing" in APP
    assert 'data-open-case="' in APP
    assert 'class="row-actions"><span class="sr-only">Открыть</span>' in HTML
    pin = CSS.split("#page-documents td.row-actions", 1)[1][:500]
    assert "display: table-cell" in pin
    assert "right: 0" in pin
    assert "min-width: 7.5rem" in pin
    assert "applyPeriodPreset(button.getAttribute" in APP
