"""R2: Plan ↔ KP concordance is visible on the case review screen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_kp_concordance_section_exists() -> None:
    assert "function renderKpConcordance(" in APP
    assert "План ↔ КП" in APP
    assert '"evidence-kp-plan"' in APP
    assert "data-focus-clinical" in APP.split("function renderKpConcordance", 1)[1][:2500]


def test_concordance_paints_after_suggest() -> None:
    assert "paintKpConcordance(" in APP
    assert "suggest.kp_concordance" in APP
    assert "bindClinicalFocusButtons(" in APP


def test_accordion_opens_plan_when_zone2b_bad() -> None:
    assert 'if (z2b === "bad") return "evidence-kp-plan"' in APP


def test_concordance_table_css() -> None:
    assert ".kp-concordance-table" in CSS
    assert ".kp-concordance-row" in CSS
