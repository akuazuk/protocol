"""W5: inspector paints chrome from the row and docks as split view at 1440."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend/web/shared/mo-ui.css").read_text(encoding="utf-8")


def test_inspector_paints_header_before_detail() -> None:
    assert "function paintCaseChrome(item)" in APP
    assert "function markOpenCaseRow(id)" in APP
    assert "function setInspecting(on)" in APP
    assert "function wideInspector()" in APP
    assert 'paintCaseChrome(state.caseNavRows[id]' in APP
    assert "Загружаем текст МО…" in APP
    assert '"пациент " + (item.patientId' not in APP
    assert "state.caseNavRows" in APP


def test_inspector_split_view_css() -> None:
    assert "body.is-inspecting .app" in CSS
    assert "min-width: 1440px" in CSS
    assert "tr[data-case].is-open" in CSS
    assert ".case-inspector-pending" in CSS
    assert 'matchMedia("(min-width: 1440px)")' in APP


if __name__ == "__main__":
    test_inspector_paints_header_before_detail()
    test_inspector_split_view_css()
    print("ok")
