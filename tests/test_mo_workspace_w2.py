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
    assert "min-width: 600px" not in CSS
    wide_1280 = CSS.split("@media (min-width: 1280px)", 1)[1][:400]
    assert "min-width: 0" in wide_1280
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
    assert "Почему так" in APP
    assert 'id="case-why"' in APP
    assert 'id="protocol-suggest-host"' in APP
    assert "Черновик модели - не меняет оценку склада" in APP
    assert 'id="case-more-details"' in APP
    assert "renderFindingsCompact(findings, crm, llmJudge, assessment)" in APP
    assert "renderClinicalDocument(sourceDocument, findings)" in APP
    assert "Подробнее: история, протокол, №55" not in APP.split("if (useZonesUi)", 1)[1].split("} else {", 1)[0]
