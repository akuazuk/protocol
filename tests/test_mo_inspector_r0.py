"""R0: case review first screen is verdict + protocol + evidence, not nested details."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_protocol_strip_is_outside_model_draft() -> None:
    zones_chunk = APP.split("if (useZonesUi)", 1)[1].split("} else {", 1)[0]
    assert 'id="protocol-suggest-host"' in zones_chunk
    assert "renderCaseWhy(zones, assessment, data.reg55)" in zones_chunk
    assert "renderEvidenceAccordion(data, history, zones)" in zones_chunk
    assert "Черновик модели - не меняет оценку склада" in zones_chunk
    assert zones_chunk.find("protocol-suggest-host") < zones_chunk.find("case-more-details")
    assert "llm-inline" not in APP


def test_wide_layout_keeps_split_until_1100() -> None:
    assert "minmax(360px, 1.2fr) minmax(400px, 1fr)" in CSS
    assert "@media (max-width: 1099px)" in CSS
    assert ".case-workspace-tabs {\n  display: none;" in CSS


def test_evidence_accordion_covers_hist_lab_meds_reg55() -> None:
    assert '"evidence-hist"' in APP
    assert '"evidence-lab"' in APP
    assert '"evidence-meds"' in APP
    assert '"evidence-reg55"' in APP
    assert '"evidence-criteria"' in APP
    assert "finding.shadow || finding.is_shadow" in APP
