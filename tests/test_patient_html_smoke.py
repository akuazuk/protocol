"""Smoke checks for patient.html v2 UI hooks."""
from __future__ import annotations

from pathlib import Path


def test_patient_html_has_v2_blocks() -> None:
    html = (Path(__file__).resolve().parents[1] / "patient.html").read_text(encoding="utf-8")
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


def test_patient_ui_has_normalize_and_render() -> None:
    js = (Path(__file__).resolve().parents[1] / "patient-ui.js").read_text(encoding="utf-8")
    for fn in (
        "normalizePatientReport",
        "renderTopSummary",
        "renderScoreCards",
        "renderClarificationPoints",
        "renderMessageToDoctor",
        "renderVisitSheet",
    ):
        assert fn in js, f"missing {fn}"
