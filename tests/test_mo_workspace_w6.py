"""W6: remaining buttons use warehouse chips, not dead duplicates."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.mo_backend import _describe_empty_state

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_period_nav_is_hidden_duplicate() -> None:
    assert 'data-page="overview" hidden' in HTML
    assert 'if (page === "overview") page = "yesterday";' in APP
    assert "page=overview открывает Обзор" in (
        ROOT / "tests" / "e2e" / "mo-smoke.spec.ts"
    ).read_text(encoding="utf-8")


def test_doctor_open_uses_overall_grade_not_zone_only() -> None:
    assert "function doctorOpenFilters()" in APP
    assert 'overallGrade: "good"' in APP
    assert 'overallGrade: "poor|important|critical"' in APP
    assert "function openDoctorCases(item, zoneKey)" in APP


def test_queue_band_drill_stays_but_only_critical_button_is_gone() -> None:
    assert "function applyQueueBand(band, opts)" in APP
    assert "state.overallGrade = \"\";" in APP
    # Волна J: кнопки «Только критические» нет, полоса очереди ставится только с KPI.
    assert "queue-critical-only" not in APP
    assert 'applyQueueBand("critical", { page: "queue" })' in APP


def test_family_clicks_set_finding_codes() -> None:
    assert "function navigateFinding(code, sourceLabel)" in APP
    assert "findingCode: code || \"\"" in APP
    assert "function navigateFamilyCode(family, codes, sourceLabel)" in APP


def test_column_checkboxes_visible_in_manager() -> None:
    # Волна J: чекбоксы колонок видны всегда (без <details>), пресеты - основной контрол.
    assert '<div class="filter-options column-all" aria-label="Отдельные колонки">' in APP
    assert 'presetButton("work"' in APP
    assert 'presetButton("review"' in APP


def test_finding_empty_state() -> None:
    empty = _describe_empty_state(
        total_records=10,
        filtered_records=0,
        params={"finding_codes": "lab_unused"},
    )
    assert empty["reason_code"] == "finding_miss"
    assert "замечанием" in empty["title"].lower()
    queue = _describe_empty_state(
        total_records=10,
        filtered_records=0,
        params={"queue_band": "critical"},
    )
    assert queue["reason_code"] == "queue_band_miss"
