"""D0: methodist decision spoiler starts collapsed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_decision_details_are_not_open_by_default() -> None:
    html = APP.split("var decisionHtml =", 1)[1][:1800]
    assert "methodist-decision-panel--dock" in html
    assert "Решение методиста" in html
    assert "--dock\" open" not in html
    assert "--dock\" open" not in APP


def test_dock_sticky_beats_generic_panel_static() -> None:
    assert ".case-workspace-decision .methodist-decision-panel.methodist-decision-panel--dock" in CSS
    assert CSS.count("position: sticky") >= 2
