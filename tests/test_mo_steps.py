"""Шаги разбора - opt-in (?steps=1); по умолчанию #case-review-pane."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (ROOT / "frontend/web/shared/mo-steps.js").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")


def test_steps_module_has_host_and_five_panels() -> None:
    assert "id=\"case-stepper\"" in STEPS
    assert "data-step-panel=\"1\"" in STEPS
    assert "data-step-panel=\"2\"" in STEPS
    assert "data-step-panel=\"3\"" in STEPS
    assert "data-step-panel=\"4\"" in STEPS
    assert "data-step-panel=\"5\"" in STEPS
    assert "proof" in STEPS
    assert "case-review-pane" not in STEPS


def test_steps_default_off_opt_in() -> None:
    block = STEPS.split("function enabled()")[1].split("function currentStep")[0]
    assert 'params.get("steps") === "1"' in block
    assert "return false;" in block


def test_app_uses_stepper_when_enabled() -> None:
    assert "MO.steps.enabled()" in APP
    assert "case-stepper-host" in APP
    assert "case-decision-dock" in APP
    assert 'id="case-review-pane"' in APP


def test_review_tab_and_zones_hero_are_default() -> None:
    assert 'data-case-tab="review">Проверка</button>' in APP
    assert 'id="case-tab-review"' in APP
    assert 'aria-selected="true" aria-controls="case-review-column"' in APP
    assert 'activateCaseWorkspaceTab("review")' in APP
    assert "renderZonesHero(zones)" in APP
    assert "renderCaseWhy(" in APP
    assert "renderFindingsCompact(" in APP
    assert "renderPassportStrip(data)" in APP
    assert 'id="case-passport-strip"' in APP


def test_html_loads_steps_before_app() -> None:
    assert HTML.index("/mo-steps.js") < HTML.index("/mo-app.js")


def test_patient_page_opens_step_one() -> None:
    assert 'data-page="patient"' in HTML
    assert 'id="patient-resolve-form"' in HTML
    assert "resolvePatientPassport" in APP
    assert "MO.steps.setStep(1)" in APP
    assert '["step", "proof", "lens", "lab_date"]' in APP
