"""W2: case review is a full-page workspace, not a 680px split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")


def test_inspector_hides_list_instead_of_squeezing() -> None:
    assert "body.is-inspecting .workspace" in CSS
    assert "body.is-inspecting .context-bar" in CSS
    assert "min(48vw, 680px)" not in CSS
    assert "margin-right: min(48vw, 680px)" not in CSS
    assert "min-width: 600px" in CSS
    assert "wideInspector() {\n      return false;" in APP


def test_open_case_is_shareable_and_back_closes() -> None:
    assert "function caseIdFromLocation(q)" in APP
    assert 'q.set("open", state.openCaseId)' in APP
    assert "closeDrawer(true, true)" in APP
    assert "if (state.pendingOpenId) openCase(state.pendingOpenId)" in APP
    assert 'id="drawer-close">К списку</button>' in HTML
    assert "period-select-legacy" in HTML
    assert 'aria-hidden="true"' in HTML


def test_first_screen_is_scores_findings_document_decision() -> None:
    assert ">Что не так<" in APP
    assert "Подробнее: история, протокол, №55" in APP
    assert 'id="case-more-details"' in APP
    assert "renderFindingsCompact(findings, crm, llmJudge, assessment)" in APP
    assert "renderClinicalDocument(sourceDocument, findings)" in APP
