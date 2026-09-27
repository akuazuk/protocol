"""F3: /doctors-dashboard - рейтинг, матрица зон, scatter, профиль выбранного врача."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from rag_server import api_methodist_mo_doctors_dashboard
from test_mo_overview_dashboard_f1 import BASE, _seed


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "doctors_dash.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


def test_endpoint_declares_period_and_facets() -> None:
    names = set(inspect.signature(api_methodist_mo_doctors_dashboard).parameters)
    for key in ("period", "date_from", "date_to", "specializations", "filials", "doctors"):
        assert key in names


def test_blocks_present_and_two_doctors(warehouse: Path) -> None:
    out = mo_backend.build_doctors_dashboard({**BASE, "document_kinds": "clinical_visit"})
    assert out["ok"] and out["available"]
    labels = sorted(item["label"] for item in out["ranking"])
    assert labels == ["Врач А", "Врач Б"]
    assert all(item["n"] == 20 and item["enough"] for item in out["ranking"])
    assert len(out["heatmap"]["rows"]) == 2
    assert [z["id"] for z in out["heatmap"]["zones"]] == ["zone1", "zone2a", "zone2b"]
    assert out["scatter"] and all(p["n"] == 20 for p in out["scatter"])
    selected = out["selected"]
    assert selected and selected["label"] in labels
    assert selected["trend"]
    assert selected["radar"] and {r["id"] for r in selected["radar"]} == {"zone1", "zone2a", "zone2b"}


def test_specialty_series_is_median_not_mean(warehouse: Path) -> None:
    out = mo_backend.build_doctors_dashboard({**BASE, "document_kinds": "clinical_visit", "doctors": "Врач А"})
    series = out["selected"]["specialty_median"]
    assert series
    # В фикстуре у специальности две точки на часть недель: медиана обязана быть числом, не AVG-алиасом.
    assert all("zone1_avg" in row for row in series)
    assert any(row["zone1_avg"] is not None for row in series)


def test_doctor_filter_selects_profile(warehouse: Path) -> None:
    out = mo_backend.build_doctors_dashboard(
        {**BASE, "document_kinds": "clinical_visit", "doctors": "Врач Б"}
    )
    assert out["selected"]["label"] == "Врач Б"
    assert out["selected"]["n"] == 20
    assert [item["label"] for item in out["ranking"]] == ["Врач Б"]
