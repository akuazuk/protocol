"""D1-D2: страницы Лекарства/Анализы и полоски на Сегодня/Период."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_backend import build_mo_capabilities, build_mo_drugs_labs_kpis
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")


def test_menu_has_medications_and_labs() -> None:
    nav = HTML.split('id="app-nav"')[1].split("</ul>")[0]
    assert 'data-page="medications"' in nav
    assert 'data-page="labs"' in nav
    assert "Лекарства" in nav
    assert "Анализы" in nav
    doctors = nav.find('data-page="doctors"')
    meds = nav.find('data-page="medications"')
    labs = nav.find('data-page="labs"')
    reports = nav.find('data-page="reports"')
    assert doctors < meds < labs < reports


def test_pages_and_strips_exist() -> None:
    assert 'id="page-medications"' in HTML
    assert 'id="page-labs"' in HTML
    assert 'id="medications-kpis"' in HTML
    assert 'id="labs-kpis"' in HTML
    assert 'id="labs-coverage"' in HTML
    assert "черновик, не в общей оценке" in HTML
    assert 'id="month-family-strip"' in HTML
    assert 'id="yesterday-family-strip"' in HTML
    overview = HTML.split('id="page-overview"')[1].split('id="page-yesterday"')[0]
    assert "month-attention" in overview
    assert "month-family-strip" in overview


def test_app_wires_family_dashboards() -> None:
    assert 'medications: "Лекарства"' in APP
    assert 'labs: "Анализы"' in APP
    assert 'loadFamilyDashboard("drug")' in APP
    assert 'loadFamilyDashboard("lab")' in APP
    assert "navigateFamilyCode" in APP
    assert "finding_family" in APP
    assert "renderFamilyScores" in APP
    assert 'switchPage(fam === "lab" ? "labs" : "medications")' in APP


def test_capabilities_hide_from_expert() -> None:
    expert = build_mo_capabilities("expert")
    methodist = build_mo_capabilities("methodist")
    assert expert["pages"]["medications"] is False
    assert expert["pages"]["labs"] is False
    assert methodist["pages"]["medications"] is True
    assert methodist["pages"]["labs"] is True


def test_drugs_labs_kpis_uses_doctor_fio_not_doctor_name(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    initialize_warehouse(db)
    doctor = doctor_key_for("Тестовый Врач")
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(fact_mo_case)")}
        assert "doctor_name" not in cols
        conn.execute(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            (doctor, "Тестовый Врач", "Терапия", "Центр"),
        )
        conn.execute(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "9101",
                "92001",
                "2026-09-01",
                "clinical_visit",
                61.0,
                "review",
                doctor,
                "Терапия",
                "Центр",
                "J06.9",
                "Болезни органов дыхания",
                "hash-kpi",
                "2026-09-01T00:00:00Z",
            ),
        )
        conn.execute(
            """INSERT INTO fact_mo_finding
               (mis_id,finding_code,severity,passed,evidence,source_ref)
               VALUES(?,?,?,?,?,?)""",
            ("9101", "C_ddi", "P1", 0, "взаимодействие", "тест"),
        )
        conn.commit()
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    payload = build_mo_drugs_labs_kpis({"date_from": "2026-09-01", "date_to": "2026-09-01"})
    assert payload["ok"] is True
    drug = payload["families"]["drug"]
    assert drug["tiles"]
    assert any(tile.get("id") == "interactions" and tile.get("cases") == 1 for tile in drug["tiles"])


def test_dockerfile_ships_finding_families() -> None:
    text = (ROOT / "deploy/gcp-app/Dockerfile").read_text(encoding="utf-8")
    assert "data/mo_finding_families/" in text
    assert (ROOT / "data/mo_finding_families/families_v1.json").is_file()
