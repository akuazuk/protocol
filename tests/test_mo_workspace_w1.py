"""W1: calendar in header, dates always apply, warehouse tables without col-filters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_header_exposes_calendar_without_apply_buttons() -> None:
    assert 'id="period-strip"' in HTML
    assert 'id="date-from-wrap"' in HTML
    assert 'hidden><span>С даты</span>' not in HTML
    assert 'id="filters-apply"' not in HTML
    assert 'id="filters-cancel"' not in HTML
    assert 'id="reset-filters"' in HTML
    assert "Сбросить всё" in HTML
    assert 'placeholder="Врач, МКБ, visit_id"' in HTML


def test_query_always_sends_date_window() -> None:
    assert "function applyPeriodPreset(period, opts)" in APP
    assert 'q.set("date_from", state.dateFrom)' in APP
    assert "if (!st.serverSort)" in APP
    assert "Среди строк на экране" in APP
    assert "Код не найден" in APP
    assert "chip-reset-all" in APP
    assert "applyPeriodPreset(button.getAttribute" in APP
