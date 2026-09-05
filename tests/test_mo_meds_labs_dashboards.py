"""D1-D2: страницы Лекарства/Анализы и полоски на Сегодня/Период."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.mo_backend import build_mo_capabilities

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
