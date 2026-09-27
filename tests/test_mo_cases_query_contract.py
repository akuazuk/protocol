"""Контракт фильтров списка МО: overall_grade и icd реально режут выборку."""
from __future__ import annotations

import inspect

from clinical_knowledge.mo_backend import _filter_records
from rag_server import api_methodist_mo_cases, api_methodist_mo_drugs_labs_kpis


def _row(case_id: str, **fields: object) -> dict:
    rec = {
        "case_id": case_id,
        "visit_id": case_id,
        "date": "2026-09-01",
        "document_kind": "clinical_visit",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "na",
        "zone2b_kp_status": "unmatched",
        "attention_primary": "none",
        "diagnosis_code": "",
        "mkb_code_main": "",
    }
    rec.update(fields)
    return rec


def test_cases_endpoint_declares_overall_grade_and_icd() -> None:
    names = set(inspect.signature(api_methodist_mo_cases).parameters)
    assert "overall_grade" in names
    assert "icd_visit_status" in names
    assert "icd" in names
    assert "queue_band" in names
    kpi_names = set(inspect.signature(api_methodist_mo_drugs_labs_kpis).parameters)
    assert "overall_grade" in kpi_names
    assert "finding_family" in kpi_names


def test_filter_records_overall_grade_good_vs_important() -> None:
    good = _row("g")
    important = _row("i", zone2a_band="bad")
    kept_good = _filter_records([good, important], {"overall_grade": "good"})
    kept_important = _filter_records([good, important], {"overall_grade": "important"})
    assert [row["case_id"] for row in kept_good] == ["g"]
    assert [row["case_id"] for row in kept_important] == ["i"]


def test_filter_records_overall_grade_pipe_multi() -> None:
    good = _row("g")
    important = _row("i", zone2a_band="bad")
    kept = _filter_records([good, important], {"overall_grade": "good|important"})
    assert {row["case_id"] for row in kept} == {"g", "i"}


def test_filter_records_icd_prefix() -> None:
    i10 = _row("a", diagnosis_code="I10", mkb_code_main="I10")
    j06 = _row("b", diagnosis_code="J06.9", mkb_code_main="J06.9")
    assert [row["case_id"] for row in _filter_records([i10, j06], {"icd": "I10"})] == ["a"]
    assert [row["case_id"] for row in _filter_records([i10, j06], {"icd": "J06"})] == ["b"]


def test_filter_records_queue_band_critical_not_overall_grade() -> None:
    critical = _row(
        "c",
        finding_codes=["C_red_flag"],
        _findings=[{"finding_code": "C_red_flag", "severity": "P0"}],
    )
    important = _row(
        "i",
        finding_codes=["B_dx_no_support"],
        _findings=[{"finding_code": "B_dx_no_support", "severity": "P1"}],
    )
    kept = _filter_records([critical, important], {"queue_band": "critical"})
    assert [row["case_id"] for row in kept] == ["c"]
    kept_imp = _filter_records([critical, important], {"queue_band": "important"})
    assert [row["case_id"] for row in kept_imp] == ["i"]
    by_visits = _filter_records(
        [critical, important],
        {"queue_band": "critical", "_queue_band_visits": {"c"}},
    )
    assert [row["case_id"] for row in by_visits] == ["c"]


# --- Волна J: каждый чип UI меняет total на складе-фикстуре -----------------------------

import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from clinical_knowledge import mo_backend  # noqa: E402
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse  # noqa: E402

_PERIOD = {"period": "custom", "date_from": "2026-07-01", "date_to": "2026-07-31", "page_size": "200"}


def _seed_chip_warehouse(path: Path) -> None:
    initialize_warehouse(path)
    doctor_a = doctor_key_for("Врач А")
    doctor_b = doctor_key_for("Врач Б")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [(doctor_a, "Врач А", "Терапия", "Центр"), (doctor_b, "Врач Б", "Кардиология", "Юг")],
        )
        for index in range(30):
            mis_id = f"m{index}"
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,doctor_key,
                    specialty,filial,diagnosis_code,diagnosis_text,icd_chapter,content_hash,updated_at,
                    zone1_band,zone2a_band,zone2b_band,zone2b_kp_status,history_tier,history_prior_n,
                    attention_primary,overall_grade)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    str(1000 + index),
                    f"2026-07-{1 + index % 20:02d}",
                    "clinical_visit",
                    float(40 + index * 2),
                    "ok",
                    doctor_a if index % 2 else doctor_b,
                    "Терапия" if index % 2 else "Кардиология",
                    "Центр" if index % 2 else "Юг",
                    "I10" if index % 3 else "J06.9",
                    "Гипертензия" if index % 3 else "ОРВИ",
                    "IX" if index % 3 else "X",
                    str(index),
                    "2026-07-31T00:00:00Z",
                    "bad" if index % 5 == 0 else "ok",
                    "bad" if index % 7 == 0 else "ok",
                    "ok" if index % 4 else "na",
                    "matched" if index % 4 else "unmatched",
                    "first_contact" if index % 6 == 0 else "known_to_doctor",
                    0 if index % 6 == 0 else 3,
                    "safety" if index % 10 == 0 else "none",
                    # Скорер записал critical только для части safety-случаев: SQL-оценка
                    # обязана уважать записанное значение, иначе чип «Критично» всегда пуст.
                    "critical" if index % 10 == 0 and index % 20 == 0 else None,
                ),
            )
            if index % 4 == 0:
                conn.execute(
                    "INSERT INTO fact_mo_finding(mis_id,finding_code,severity,passed,evidence,source_ref) VALUES(?,?,?,?,?,?)",
                    (mis_id, "C_red_flag", "P0", 0, "", "protocol:55:1"),
                )
            if index % 5 == 0:
                conn.execute(
                    "INSERT INTO fact_mo_finding(mis_id,finding_code,severity,passed,evidence,source_ref) VALUES(?,?,?,?,?,?)",
                    (mis_id, "B_lab_unused_in_plan", "P2", 0, "", "protocol:55:1"),
                )
        conn.executescript(mo_backend.CRM_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO crm_case_state(case_id,status,updated_at,updated_by) VALUES(?,?,?,?)",
            [
                ("1000", "closed", "2026-07-31T00:00:00Z", "test"),
                ("1001", "in_review", "2026-07-31T00:00:00Z", "test"),
                ("1002", "in_review", "2026-07-31T00:00:00Z", "test"),
            ],
        )
        conn.commit()


@pytest.fixture()
def chip_warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "chips.sqlite"
    _seed_chip_warehouse(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


# Все чипы, которые UI (mo-app.js query()) умеет ставить в запрос /cases.
UI_CHIPS = [
    {"overall_grade": "critical"},
    {"overall_grade": "critical|important|poor"},
    {"doctors": "Врач А"},
    {"specializations": "Терапия"},
    {"filials": "Юг"},
    {"icd": "I10"},
    {"mkb_chapters": "X"},
    {"q": "ОРВИ"},
    {"queue_only": "1"},
    {"finding_family": "lab"},
    {"finding_codes": "C_red_flag"},
    {"zone": "zone1", "zone_band": "bad"},
    {"zone": "zone2a", "zone_band": "bad"},
    {"zone_band": "bad"},
    {"kp_status": "unmatched"},
    {"history_tier": "first_contact"},
    {"attention_only": "1"},
    {"crm_statuses": "in_review"},
    {"crm_statuses": "closed|in_review"},
    {"queue_band": "critical"},
]


@pytest.mark.parametrize("chip", UI_CHIPS, ids=[",".join(c) for c in UI_CHIPS])
def test_every_ui_chip_changes_total(chip_warehouse: Path, chip: dict[str, str]) -> None:
    base = mo_backend.build_cases(dict(_PERIOD))
    assert base["total"] == 30
    result = mo_backend.build_cases({**_PERIOD, **chip})
    assert result["total"] != base["total"], chip
    assert result["total"] > 0, chip
    assert result["total"] == len(result["rows"]), chip


def test_stored_critical_grade_wins_over_zone_approximation(chip_warehouse: Path) -> None:
    critical = mo_backend.build_cases({**_PERIOD, "overall_grade": "critical"})
    assert sorted(str(r["case_id"]) for r in critical["rows"]) == ["1000", "1020"]
    important = mo_backend.build_cases({**_PERIOD, "overall_grade": "important"})
    ids = {str(r["case_id"]) for r in important["rows"]}
    assert "1010" in ids and "1000" not in ids
    for row in critical["rows"]:
        assert row["overall_grade"]["grade"] == "critical"


def test_crm_statuses_stays_sql_pageable_and_matches_python(chip_warehouse: Path) -> None:
    params = {**_PERIOD, "crm_statuses": "in_review"}
    assert mo_backend._cases_sql_pageable(params)
    result = mo_backend.build_cases(dict(params))
    assert sorted(str(r["case_id"]) for r in result["rows"]) == ["1001", "1002"]
    assert result["total"] == 2
    # «new» = записи без строки CRM: их большинство, и total это отражает.
    fresh = mo_backend.build_cases({**_PERIOD, "crm_statuses": "new"})
    assert fresh["total"] == 27
    where, values = mo_backend._warehouse_where({**_PERIOD, "crm_statuses": "closed"})
    assert any("crm_case_state" in clause for clause in where)
    assert "closed" in values


def test_facets_expose_crm_statuses_with_counts(chip_warehouse: Path) -> None:
    facets = mo_backend.build_facets(dict(_PERIOD))["facets"]
    by_value = {item["value"]: item for item in facets["crm_statuses"]}
    assert set(by_value) >= {"new", "in_review", "closed"}
    assert by_value["new"]["n"] == 27
