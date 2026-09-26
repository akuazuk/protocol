"""Волна B: SQL-пейджинг для сортировок, замечаний и очереди; freshness без чтения строк."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse

PERIOD = {"period": "custom", "date_from": "2026-07-01", "date_to": "2026-07-31"}


def _seed(path: Path) -> None:
    initialize_warehouse(path)
    doctor_a = doctor_key_for("Врач А")
    doctor_b = doctor_key_for("Врач Б")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [(doctor_a, "Врач А", "Терапия", "Центр"), (doctor_b, "Врач Б", "Кардиология", "Юг")],
        )
        for index in range(40):
            mis_id = f"m{index}"
            score = None if index % 10 == 9 else float(50 + index)
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,doctor_key,
                    specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at,
                    zone1_pct,zone1_band,zone2a_band,zone2b_band,zone2b_kp_status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    str(1000 + index),
                    f"2026-07-{1 + index % 20:02d}",
                    "clinical_visit",
                    score,
                    "manual_review_required" if index == 5 else "ok",
                    doctor_a if index % 2 else doctor_b,
                    "Терапия" if index % 2 else "Кардиология",
                    "Центр" if index % 2 else "Юг",
                    "I10" if index % 3 else "J06",
                    "IX" if index % 3 else "X",
                    str(index),
                    "2026-07-31T00:00:00Z",
                    float(index),
                    "bad" if index % 7 == 0 else "ok",
                    "ok",
                    "ok",
                    "matched",
                ),
            )
            findings = []
            if index % 4 == 0:
                findings.append(("S_red_flag", "P0"))
            if index % 5 == 0:
                findings.append(("L_lab_missing", "P2"))
            if index % 6 == 0:
                findings.append(("D_drug_dose", "P1"))
            for code, severity in findings:
                conn.execute(
                    """INSERT INTO fact_mo_finding
                       (mis_id,finding_code,severity,passed,evidence,source_ref)
                       VALUES(?,?,?,?,?,?)""",
                    (mis_id, code, severity, 0, "", "protocol:55:1"),
                )
        conn.executescript(mo_backend.CRM_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO crm_case_state(case_id,status,updated_at,updated_by) VALUES(?,?,?,?)",
            ("1000", "closed", "2026-07-31T00:00:00Z", "test"),
        )
        conn.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "mo.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


def _python_reference(params: dict[str, str]) -> list[str]:
    """Тот же ответ, что давал старый путь: все строки в Python, затем сортировка."""
    full = mo_backend._filter_records(mo_backend._warehouse_records(params), params)
    states = mo_backend._crm_states([r["case_id"] for r in full])
    if str(params.get("queue_only") or "") == "1":
        full = [
            r
            for r in full
            if states.get(r["case_id"], {}).get("status", "new")
            not in {"false_positive", "resolved", "closed"}
        ]
    return sorted(r["case_id"] for r in full)


def test_sql_pageable_now_covers_sort_findings_and_queue(warehouse: Path) -> None:
    assert mo_backend._cases_sql_pageable({**PERIOD, "sort_by": "overall"})
    assert mo_backend._cases_sql_pageable({**PERIOD, "sort_by": "overall_grade"})
    assert mo_backend._cases_sql_pageable({**PERIOD, "sort_by": "doctor", "sort_dir": "asc"})
    assert mo_backend._cases_sql_pageable({**PERIOD, "finding_family": "lab"})
    assert mo_backend._cases_sql_pageable({**PERIOD, "finding_codes": "S_red_flag"})
    assert mo_backend._cases_sql_pageable({**PERIOD, "queue_only": "1"})
    # Что не выражается складом - по-прежнему Python.
    assert not mo_backend._cases_sql_pageable({**PERIOD, "queue_band": "critical"})
    assert not mo_backend._cases_sql_pageable({**PERIOD, "sort_by": "patient_id"})


@pytest.mark.parametrize(
    "extra",
    [
        {"finding_family": "lab"},
        {"finding_codes": "S_red_flag|D_drug_dose"},
        {"finding_family": "drug", "finding_codes": "L_lab_missing"},
        {"queue_only": "1"},
        {"statuses": "manual_review_required"},
    ],
)
def test_sql_filters_match_python_semantics(warehouse: Path, extra: dict[str, str]) -> None:
    params = {**PERIOD, **extra, "page_size": "200"}
    result = mo_backend.build_cases(dict(params))
    ids = sorted(str(row["case_id"]) for row in result["rows"])
    assert ids == _python_reference(dict(params))
    assert result["total"] == len(ids)


def test_sql_sort_by_score_nulls_last_and_direction(warehouse: Path) -> None:
    desc = mo_backend.build_cases({**PERIOD, "sort_by": "overall", "page_size": "200"})
    scores = [row["overall_pct"] for row in desc["rows"]]
    numeric = [s for s in scores if s is not None]
    assert numeric == sorted(numeric, reverse=True)
    assert scores[-len(scores) + len(numeric) :] == [None] * (len(scores) - len(numeric))
    asc = mo_backend.build_cases({**PERIOD, "sort_by": "overall", "sort_dir": "asc", "page_size": "200"})
    numeric_asc = [row["overall_pct"] for row in asc["rows"] if row["overall_pct"] is not None]
    assert numeric_asc == sorted(numeric_asc)
    assert asc["rows"][-1]["overall_pct"] is None

    by_doctor = mo_backend.build_cases({**PERIOD, "sort_by": "doctor", "sort_dir": "asc", "page_size": "200"})
    names = [row["doctor_fio"] for row in by_doctor["rows"]]
    assert names == sorted(names)


def test_sql_paging_keeps_total_and_pages_disjoint(warehouse: Path) -> None:
    first = mo_backend.build_cases({**PERIOD, "sort_by": "overall", "page": 1, "page_size": 15})
    second = mo_backend.build_cases({**PERIOD, "sort_by": "overall", "page": 2, "page_size": 15})
    assert first["total"] == second["total"] == 40
    assert len(first["rows"]) == 15 and len(second["rows"]) == 15
    assert not {r["case_id"] for r in first["rows"]} & {r["case_id"] for r in second["rows"]}


def test_freshness_uses_sql_counts_and_cache_invalidates_on_write(warehouse: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_RESULT_CACHE", "1")
    mo_backend._result_cache_clear()
    calls = {"n": 0}
    original = mo_backend._freshness_counts_sql

    def counting(params):
        calls["n"] += 1
        return original(params)

    monkeypatch.setattr(mo_backend, "_freshness_counts_sql", counting)
    first = mo_backend.build_freshness(dict(PERIOD))
    assert first["total_records"] == 40
    assert first["data_through"] == "2026-07-20"
    assert first["filtered_records"] == 40
    again = mo_backend.build_freshness(dict(PERIOD))
    assert again["total_records"] == 40
    assert calls["n"] == 1, "второй вызов должен прийти из кэша"

    filtered = mo_backend.build_freshness({**PERIOD, "statuses": "manual_review_required"})
    assert filtered["filtered_records"] == 1
    assert filtered["total_records"] == 40

    with sqlite3.connect(warehouse) as conn:
        conn.execute("DELETE FROM fact_mo_case WHERE mis_id = 'm0'")
        conn.commit()
    after = mo_backend.build_freshness(dict(PERIOD))
    assert after["total_records"] == 39, "запись в витрину меняет отпечаток и сбрасывает кэш"


def test_facets_light_records_have_same_counts(warehouse: Path) -> None:
    facets = mo_backend.build_facets(dict(PERIOD))
    assert facets["n_filtered"] == 40
    specialties = {item["value"]: item["n"] for item in facets["facets"]["specialties"]}
    assert specialties == {"Терапия": 20, "Кардиология": 20}
    light = mo_backend._records(dict(PERIOD), light=True)
    assert light and light[0]["assessment"].get("light") is True
    assert "_reg55_weak_points_json" not in mo_backend._public_row(light[0])


def test_facets_sql_matches_row_path(warehouse: Path, monkeypatch) -> None:
    params = {**PERIOD, "document_kinds": "clinical_visit", "score_eligible_only": "1"}
    assert mo_backend._facets_sql_supported(params)
    sql = mo_backend._build_facets_sql(dict(params))
    monkeypatch.setattr(mo_backend, "_facets_sql_supported", lambda _p: False)
    rows = mo_backend._build_facets_uncached(dict(params))
    assert sql["n_filtered"] == rows["n_filtered"] == 40
    for key in ("specialties", "filials", "statuses", "score_bands"):
        assert {(i["value"], i.get("n")) for i in sql["facets"][key]} == {
            (i["value"], i.get("n")) for i in rows["facets"][key]
        }, key
    assert {(i["key"], i["n"]) for i in sql["facets"]["mkb_chapters"]} == {
        (i["key"], i["n"]) for i in rows["facets"]["mkb_chapters"]
    }
    assert [(i["value"], i["n"]) for i in sql["facets"]["doctors"]] == [
        (i["value"], i["n"]) for i in rows["facets"]["doctors"]
    ]
    assert {(i["value"], i["n"]) for i in sql["facets"]["document_kinds"]} == {
        (i["value"], i["n"]) for i in rows["facets"]["document_kinds"]
    }
    assert {(i["value"], i["n"]) for i in sql["facets"]["crm_statuses"]} == {
        (i["value"], i["n"]) for i in rows["facets"]["crm_statuses"]
    }
    # Фильтр, которого нет в SQL (queue_band) - честный откат на строки.
    assert not mo_backend._facets_sql_supported({**params, "queue_band": "critical"})
    # exclude_* и mkb_chapters теперь в SQL: считаются одинаково
    narrowed = {**params, "exclude_filials": "Юг", "mkb_chapters": "IX"}
    sql_n = mo_backend._build_facets_sql(dict(narrowed))
    rows_n = mo_backend._build_facets_uncached(dict(narrowed))
    assert sql_n["n_filtered"] == rows_n["n_filtered"] > 0
