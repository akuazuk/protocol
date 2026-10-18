"""Волна C: контракт фильтров на крошечной sqlite, без живого склада в CI."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "tests" / "fixtures" / "mo_filter_contract_window.json"
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")


def _seed(db: Path) -> None:
    initialize_warehouse(db)
    hyphen = doctor_key_for("Альфа-Бета Тест")
    plain = doctor_key_for("Гамма Тест")
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key, doctor_fio, specialty, filial) VALUES(?,?,?,?)",
            [
                (hyphen, "Альфа-Бета Тест", "Терапевт", "Филиал А"),
                (plain, "Гамма Тест", "Терапевт", "Филиал А"),
            ],
        )
        rows = [
            ("c-good", "v-good", "clinical_visit", "good", hyphen, "J06.9"),
            ("c-important", "v-important", "clinical_visit", "important", hyphen, "I10"),
            ("c-consult", "v-consult", "consultation", "good", hyphen, "J06.9"),
        ]
        conn.executemany(
            """INSERT INTO fact_mo_case
               (mis_id, visit_id, visit_date, document_kind, overall_grade, status,
                doctor_key, specialty, filial, diagnosis_code, content_hash, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (mis, visit, "2026-10-02", kind, grade, "review", doctor, "Терапевт", "Филиал А", icd, mis, "2026-10-02T00:00:00Z")
                for mis, visit, kind, grade, doctor, icd in rows
            ],
        )
        conn.commit()


def _cases(monkeypatch, tmp_path: Path, **extra: str) -> dict:
    db = tmp_path / "mo.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    mo_backend._result_cache_clear()
    params = {
        "date_from": "2026-10-01",
        "date_to": "2026-10-17",
        "document_kinds": "clinical_visit",
        "score_eligible_only": "1",
        "page": "1",
        "page_size": "20",
    }
    params.update(extra)
    return mo_backend.build_cases(params)


def _ids(payload: dict) -> set[str]:
    rows = payload.get("rows") or payload.get("cases") or payload.get("items") or []
    return {str(row.get("visit_id") or row.get("id") or "") for row in rows}


def test_window_fixture_is_not_a_live_query() -> None:
    data = json.loads(WINDOW.read_text(encoding="utf-8"))
    grades = data["grades"]
    assert grades["good"] + grades["fair"] + grades["poor"] + grades["important"] + grades["critical"] == grades["all"]
    assert data["window"]["ci"].startswith("не сравнивать")
    assert "gcloud" not in WINDOW.read_text(encoding="utf-8")


def test_consultation_stays_out_and_lowercase_hyphen_name_finds_clinical(monkeypatch, tmp_path: Path) -> None:
    payload = _cases(monkeypatch, tmp_path, q="альфа-бета тест")
    found = _ids(payload)
    assert "v-consult" not in found
    assert "v-good" in found or "v-important" in found
    assert int(payload.get("total") or payload.get("n") or len(found)) >= 1


def test_critical_chip_is_zero_when_fixture_has_no_stored_critical(monkeypatch, tmp_path: Path) -> None:
    payload = _cases(monkeypatch, tmp_path, overall_grade="critical")
    assert _ids(payload) == set() or _ids(payload) == {""}
    assert int(payload.get("total") or payload.get("n") or 0) == 0


def test_important_chip_returns_only_that_grade(monkeypatch, tmp_path: Path) -> None:
    payload = _cases(monkeypatch, tmp_path, overall_grade="important")
    assert "v-important" in _ids(payload)
    assert "v-good" not in _ids(payload)
    assert "v-consult" not in _ids(payload)


def test_wave_b_controls_match_the_click_contract() -> None:
    assert 'applyPeriodPreset("yesterday")' in APP
    assert 'applyPeriodPreset("7d")' in APP
    assert 'applyPeriodPreset("month")' in APP
    assert "Очередь: только критичное замечание" in APP
    assert "keepGrade: true" in APP
    assert "Слабо, важно и критично" in HTML
    assert 'overallGrade: "poor|important|critical"' in APP
    assert 'data-filter-apply>Показать' in APP
