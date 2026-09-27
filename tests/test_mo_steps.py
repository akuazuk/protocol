"""P4: шаги разбора вместо главного скролла #case-review-pane."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (ROOT / "frontend/web/shared/mo-steps.js").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")


def test_steps_module_has_host_and_three_panels() -> None:
    assert "id=\"case-stepper\"" in STEPS
    assert "data-step-panel=\"1\"" in STEPS
    assert "data-step-panel=\"2\"" in STEPS
    assert "data-step-panel=\"3\"" in STEPS
    assert "case-review-pane" not in STEPS


def test_app_uses_stepper_when_enabled() -> None:
    assert "MO.steps.enabled()" in APP
    assert "case-stepper-host" in APP
    assert 'id="case-review-pane"' in APP


def test_html_loads_steps_before_app() -> None:
    assert HTML.index("/mo-steps.js") < HTML.index("/mo-app.js")
