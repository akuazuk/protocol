"""F5: /labs-dashboard - окно лаборатории, unused-тесты, отклонения, тренд, покрытие."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from rag_server import api_methodist_mo_labs_dashboard
from test_mo_overview_dashboard_f1 import BASE, _seed


def _seed_labs(path: Path, lab_path: Path) -> None:
    _seed(path)
    with sqlite3.connect(path) as conn:
        for index in range(40):
            conn.execute(
                "UPDATE fact_mo_case SET patient_key=? WHERE mis_id=?",
                (f"labpk{index % 8:02d}", f"m08{index}"),
            )
        extra: list[tuple[str, str, str, int, str, str, str]] = []
        for index in range(0, 40, 4):
            extra.append(
                (
                    f"m08{index}",
                    "B_lab_unused_in_dx",
                    "P1",
                    0,
                    "Готовый анализ не учтён в диагнозе",
                    "Есть результаты: ОАК, глюкоза, но в диагнозе они не отражены.",
                    "ОАК, глюкоза",
                )
            )
        for index in range(0, 40, 5):
            extra.append(
                (
                    f"m08{index}",
                    "B_lab_abnormal_ignored",
                    "P1",
                    0,
                    "Отклонение анализа не отражено в заключении",
                    "Вне референса: глюкоза=8.2 (норма 3.3-6.1).",
                    "глюкоза",
                )
            )
        for index in range(0, 40, 10):
            extra.append(
                (
                    f"m08{index}",
                    "B_exams_gap",
                    "P2",
                    0,
                    "Пропуск обследования по КП",
                    "В плане нет обязательного ОАК.",
                    "ОАК",
                )
            )
        conn.executemany(
            """INSERT OR REPLACE INTO fact_mo_finding
               (mis_id,finding_code,severity,passed,title_ru,detail_ru,evidence)
               VALUES(?,?,?,?,?,?,?)""",
            extra,
        )
        conn.commit()
    with sqlite3.connect(lab_path) as lab:
        lab.execute(
            """CREATE TABLE fact_mo_lab (
                 patient_key TEXT NOT NULL,
                 test_date TEXT NOT NULL,
                 test_id INTEGER NOT NULL,
                 type_id INTEGER,
                 type_name TEXT,
                 indicator_id INTEGER,
                 indicator_name TEXT,
                 value TEXT,
                 unit TEXT
               )"""
        )
        rows = []
        for index in range(40):
            if index % 8 > 3:
                continue
            day = 1 + index % 20
            rows.append(
                (
                    f"labpk{index % 8:02d}",
                    f"2026-08-{day:02d}",
                    index,
                    20,
                    "БАК",
                    201,
                    "Глюкоза",
                    "5.4",
                    "ммоль/л",
                )
            )
        lab.executemany("INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)", rows)
        lab.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "labs_dash.sqlite"
    lab = tmp_path / "mo_lab.sqlite"
    _seed_labs(db, lab)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_LAB_DB", str(lab))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


def test_endpoint_declares_period_and_facets() -> None:
    names = set(inspect.signature(api_methodist_mo_labs_dashboard).parameters)
    for key in ("period", "date_from", "date_to", "specializations", "filials", "doctors", "finding_codes"):
        assert key in names


def test_blocks_present_and_window_splits(warehouse: Path) -> None:
    out = mo_backend.build_labs_dashboard({**BASE, "document_kinds": "clinical_visit"})
    assert out["ok"] and out["available"]
    assert out["total_cases"] == 40
    assert out["window"]["available"] is True
    assert out["window"]["has"] == 20
    assert out["window"]["none"] == 20
    assert out["window"]["unused"] >= 1
    assert {t["id"] for t in out["tiles"]} >= {"has", "unused", "abnormal"}
    labels = [t["label"] for t in out["unused_tests"]]
    assert "ОАК" in labels
    assert "глюкоза" in labels
    assert out["abnormal_specialty"]
    assert {r["specialty"] for r in out["abnormal_specialty"]} >= {"Терапия", "Кардиология"}
    assert any((r["n"] or 0) > 0 for r in out["abnormal_specialty"])
    assert out["trend"]
    months = [item["month"] for item in out["coverage_months"]["items"]]
    assert months[0] == "2025-12"
    assert "2026-08" in months
    august = next(item for item in out["coverage_months"]["items"] if item["month"] == "2026-08")
    assert august["has"] == 20
    assert august["n"] == 40


def test_facets_apply_to_every_block(warehouse: Path) -> None:
    full = mo_backend.build_labs_dashboard({**BASE, "document_kinds": "clinical_visit"})
    part = mo_backend.build_labs_dashboard(
        {**BASE, "document_kinds": "clinical_visit", "specializations": "Терапия"}
    )
    assert part["total_cases"] == 20 < full["total_cases"]
    assert [r["specialty"] for r in part["abnormal_specialty"]] == ["Терапия"]
    by_doctor = mo_backend.build_labs_dashboard(
        {**BASE, "document_kinds": "clinical_visit", "doctors": "Врач Б"}
    )
    assert by_doctor["total_cases"] == 20


def test_missing_lab_warehouse_keeps_findings(warehouse: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_LAB_DB", str(tmp_path / "missing.sqlite"))
    out = mo_backend.build_labs_dashboard({**BASE, "document_kinds": "clinical_visit"})
    assert out["ok"] and out["available"]
    assert out["window"]["available"] is False
    assert out["window"]["reason"]
    assert out["unused_tests"]
    assert out["coverage_months"]["available"] is False


def test_empty_window_has_reason(warehouse: Path) -> None:
    out = mo_backend.build_labs_dashboard(
        {"period": "custom", "date_from": "2025-01-01", "date_to": "2025-01-31"}
    )
    assert out["ok"]
    assert out["available"] is False
    assert out["reason"]
    assert out["unused_tests"] == []
