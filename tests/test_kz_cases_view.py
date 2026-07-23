"""Тесты дашборда §7Б: главы МКБ, нормализация кейса, фильтры, агрегаты, cases-view."""
from __future__ import annotations

import json

import pytest

from clinical_knowledge import mis_kz_quality as m


def test_icd10_chapter_ranges():
    assert m.icd10_chapter("H66.1")[0] == "VIII"
    assert m.icd10_chapter("H10")[0] == "VII"
    assert m.icd10_chapter("I20.0")[0] == "IX"
    assert m.icd10_chapter("D48")[0] == "II"
    assert m.icd10_chapter("D50")[0] == "III"
    assert m.icd10_chapter("")[1] == "без кода"
    assert m.icd10_chapter("J06.9")[0] == "X"


def test_age_group_and_band():
    assert m._age_group(5) == "дети (<18)"
    assert m._age_group(40) == "взрослые (18-64)"
    assert m._age_group(80) == "пожилые (65+)"
    assert m._age_group(None) == "неизв."
    assert m._score_band(30) == "<50"
    assert m._score_band(60) == "50-75"
    assert m._score_band(80) == "75-90"
    assert m._score_band(95) == "≥90"
    assert m._score_band(None) == "нет скора"


def _case(vid, overall, spec, code, p0=0, harm=False, axes_finding=None):
    return {
        "visit_id": vid,
        "patient_id": "p" + vid,
        "date": "2026-01-05",
        "doctor_fio": "Иванов И.И.",
        "doctor_specialization": spec,
        "filial": "Центр",
        "diagnosis_short": "тест  диагноз\r\n перенос",
        "overall_pct": None,
        "status": None,
        "deep": {
            "axes": {
                "documentation": 80.0,
                "clinical_concordance": 85.0,
                "safety": 100.0 if not harm else 40.0,
                "regulatory": 88.0,
            },
            "overall_pct": overall,
            "status": "good" if overall >= 75 else "acceptable",
            "n_findings": (1 if axes_finding else 0),
            "n_by_severity": {"P0": p0, "P1": 0, "P2": 0, "P3": 1},
            "has_potential_harm": harm,
            "protocol_used": True,
            "findings": (
                [{"axis": axes_finding, "severity": "P0" if p0 else "P3", "passed": False, "needs_human": harm}]
                if axes_finding else []
            ),
        },
    }


def test_flat_case_normalization():
    rec = m._flat_case(_case("1", 48.0, "ЛОР-врач", "H66.1"), {"mkb_code_main": "H66.1", "patient_age_years": "70", "kz_kind": "kz"})
    assert rec["icd_chapter"] == "VIII"
    assert rec["age_group"] == "пожилые (65+)"
    assert rec["score_band"] == "<50"
    assert rec["overall_pct"] == 48.0
    assert "\r" not in rec["diagnosis_short"] and "  " not in rec["diagnosis_short"]
    assert rec["status"] == "review"  # статус пересчитан из overall(48)+axes через risk-gate


@pytest.fixture()
def cases_file(tmp_path, monkeypatch):
    cases = [
        _case("1", 48.0, "ЛОР-врач", "H66.1", p0=1, harm=True, axes_finding="safety"),
        _case("2", 88.0, "Терапевт", "J06.9", axes_finding="documentation"),
        _case("3", 92.0, "Терапевт", "I20.0"),
    ]
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
    monkeypatch.setenv("MIS_KZ_CASES_PATH", str(p))
    # без CSV: _csv_path_for_month вернёт None -> merge пропускается
    monkeypatch.setattr(m, "_csv_path_for_month", lambda month: None)
    m._CASES_CACHE.clear()
    m._CSV_BY_VISIT_CACHE.clear()
    return p


def test_cases_view_basic(cases_file):
    v = m.build_kz_cases_view(month="2026-01", page=1, page_size=10, sort_by="overall", sort_dir="asc")
    assert v["ok"] is True
    assert v["total"] == 3
    # сортировка по overall asc -> худший первый
    assert v["rows"][0]["visit_id"] == "1"
    assert v["filtered_agg"]["severity_totals"]["P0"] == 1
    assert v["filtered_agg"]["n_potential_harm"] == 1
    axes = [f["value"] for f in v["facets"]["finding_axes"]]
    assert "safety" in axes and "documentation" in axes


def test_cases_view_filter_specialty(cases_file):
    v = m.build_kz_cases_view(month="2026-01", specialization="Терапевт")
    assert v["total"] == 2
    assert all(r["specialization"] == "Терапевт" for r in v["rows"])


def test_cases_view_preset_p0(cases_file):
    v = m.build_kz_cases_view(month="2026-01", preset="p0")
    assert v["total"] == 1
    assert v["rows"][0]["visit_id"] == "1"


def test_cases_view_score_band_filter(cases_file):
    v = m.build_kz_cases_view(month="2026-01", score_band="≥90")
    assert v["total"] == 1
    assert v["rows"][0]["visit_id"] == "3"


def test_cases_view_potential_harm_filter(cases_file):
    v = m.build_kz_cases_view(month="2026-01", potential_harm=True)
    assert v["total"] == 1
    assert v["rows"][0]["has_potential_harm"] is True
