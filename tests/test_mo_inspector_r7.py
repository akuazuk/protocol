"""R7: case review polish - sticky dock, anchors, one verdict, theme, E22."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_decision_dock_is_sticky_and_open() -> None:
    assert 'methodist-decision-panel methodist-decision-panel--dock" open' in APP
    assert ".methodist-decision-panel--dock {\n  flex: 0 0 auto;\n  position: sticky;" in CSS
    narrow = CSS.split("@media (max-width: 1099px)", 1)[1]
    assert "position: static" not in narrow.split("@media (max-width: 720px)", 1)[0]


def test_finding_and_zone_focus_keep_split_column() -> None:
    chunk = APP.split("function caseWorkspaceIsSplit(", 1)[1][:3500]
    assert 'matchMedia("(min-width: 1100px)")' in chunk
    assert 'if (!caseWorkspaceIsSplit()) activateCaseWorkspaceTab("document")' in chunk
    assert "function applyZoneCardFocus(zone)" in chunk
    assert '"evidence-criteria"' in chunk
    assert '"evidence-hist"' in chunk
    assert '"evidence-kp-plan"' in chunk
    assert 'role="button" tabindex="0"' in APP.split("function renderZonesHero(", 1)[1][:1800]


def test_first_screen_has_one_verdict_heading() -> None:
    zones = APP.split("if (useZonesUi)", 1)[1].split("} else {", 1)[0]
    assert "Черновик модели - не меняет оценку склада" in zones
    assert "renderReviewBrief" in zones
    assert "renderLlmActionJudge" in zones
    assert zones.find("case-more-details") < zones.find("renderReviewBrief")
    assert "<h3>Итог разбора</h3>" not in APP
    assert "<h3>Черновик сводки модели</h3>" in APP
    assert "Этапы модели" in APP


def test_theme_tokens_replace_hardcoded_light_surfaces() -> None:
    assert "background: #fbfcfd" not in CSS
    assert "background: #fff6f5" not in CSS
    assert "background: #e8f5e9" not in CSS
    assert "color-mix(in srgb, var(--surface-2) 65%, var(--surface-solid))" in CSS
    assert "color-mix(in srgb, var(--bad) 10%, var(--surface-solid))" in CSS
    assert "color-mix(in srgb, var(--good) 10%, var(--surface-solid))" in CSS
    assert ".zone-card:focus-visible" in CSS


def test_e22_narrow_overflow_contract() -> None:
    assert "min-width: 600px" not in CSS
    narrow = CSS.split("@media (max-width: 720px)", 1)[1][:1200]
    assert "overflow-x: hidden" in narrow
    assert "overflow-wrap: anywhere" in narrow
    assert ".drawer.case-workspace" in narrow
