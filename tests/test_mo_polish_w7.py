"""W7: table polish, column presets, reports empty, Escape closes search first."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-ui.css").read_text(encoding="utf-8")


def test_column_presets_and_reports_empty() -> None:
    assert "var COLUMN_PRESETS" in APP
    assert 'data-preset="work"' in APP
    assert 'data-preset="review"' in APP
    assert "Нет файла за дату" in APP
    assert 'id="reports-open-overview"' in APP
    assert 'switchPage("yesterday")' in APP


def test_table_zebra_and_escape_closes_suggestions_first() -> None:
    assert "#document-rows tr:nth-child(even) td" in CSS
    assert "#document-rows tr:focus-visible" in CSS
    assert ".column-presets" in CSS
    assert 'if (suggestions && !suggestions.hidden)' in APP


if __name__ == "__main__":
    test_column_presets_and_reports_empty()
    test_table_zebra_and_escape_closes_suggestions_first()
    print("ok")
