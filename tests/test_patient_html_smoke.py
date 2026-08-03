"""Smoke checks for patient.html v2 UI hooks."""
from __future__ import annotations

from pathlib import Path


PATIENT_ROOT = Path(__file__).resolve().parents[1] / "frontend" / "web" / "patient"


def test_patient_html_has_v2_blocks() -> None:
    html = (PATIENT_ROOT / "patient.html").read_text(encoding="utf-8")
    for needle in (
        "score-cards-wrap",
        "top-summary-wrap",
        "clarify-wrap",
        "message-doctor-wrap",
        "visit-sheet-wrap",
        "plain-terms-wrap",
        "Protocol",
        "Краткий итог",
    ):
        assert needle in html, f"missing {needle}"


def test_patient_ui_js_syntax_valid() -> None:
    import subprocess

    js = PATIENT_ROOT / "patient-ui.js"
    proc = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_patient_ui_has_normalize_and_render() -> None:
    js = (PATIENT_ROOT / "patient-ui.js").read_text(encoding="utf-8")
    for fn in (
        "normalizePatientReport",
        "renderTopSummary",
        "renderScoreCards",
        "renderClarificationPoints",
        "renderMessageToDoctor",
        "renderVisitSheet",
        "renderUploadJokeCard",
        "renderUploadJokeQuestions",
        "resetToUploadForm",
        "upload-joke-retry",
        "wireUploadZone",
    ):
        assert fn in js, f"missing {fn}"
