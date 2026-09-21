"""U2: case review first screen after collapsed dock is zones + protocol + why."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_first_screen_puts_protocol_before_why() -> None:
    zones = APP.split("if (useZonesUi)", 1)[1].split("} else {", 1)[0]
    assert zones.find("protocol-suggest-host") < zones.find("renderCaseWhy")
    assert zones.find("renderZonesHero") < zones.find("protocol-suggest-host")
    assert zones.find("renderCaseWhy") < zones.find("renderFindingsCompact")
    assert zones.find("renderFindingsCompact") < zones.find("renderEvidenceAccordion")
    assert "Черновик модели - не меняет оценку склада" in zones


def test_findings_and_evidence_start_folded() -> None:
    assert 'class="detail-block case-findings-fold"' in APP
    assert "var openId = pickOpenEvidenceId" not in APP.split(
        "function renderEvidenceAccordion", 1
    )[1][:400]
    assert 'var openId = "";' in APP.split("function renderEvidenceAccordion", 1)[1][:250]
    assert "function pickOpenEvidenceId" in APP
    assert "openEvidencePanel(spec.evidence)" in APP


def test_protocol_host_is_a_name_strip_not_concordance() -> None:
    chunk = APP.split("function renderProtocolSuggest(", 1)[1].split("function verdictSelect(", 1)[0]
    assert "protocol-suggest-block--strip" in chunk
    assert "protocol-suggest-top--empty" in chunk
    assert "Протокол не подобран" in chunk
    assert "+ concordanceHtml" not in chunk
    assert "paintKpConcordance" in APP


def test_collapsed_dock_is_thinner_and_wide_split_has_no_tabs() -> None:
    assert ".methodist-decision-panel--dock:not([open])" in CSS
    assert ".case-workspace-tabs {\n  display: none;" in CSS
    narrow = CSS.split("@media (max-width: 720px)", 1)[1][:900]
    assert "overflow-x: hidden" in narrow
    assert ".case-workspace-grid--zones" in narrow
