"""F1: /overview-dashboard - O1-O6 одним ответом, за окно и с фасетами, со сравнением."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse
from rag_server import api_methodist_mo_overview_dashboard


def _seed(path: Path) -> None:
    """Июль (сравнение) и август (текущий): 2 специальности, 2 филиала, разные оценки."""
    initialize_warehouse(path)
    doc_a = doctor_key_for("Врач А")
    doc_b = doctor_key_for("Врач Б")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [(doc_a, "Врач А", "Терапия", "Центр"), (doc_b, "Врач Б", "Кардиология", "Юг")],
        )
        rows = []
        for month, count in (("07", 20), ("08", 40)):
            for index in range(count):
                day = 1 + index % 20
                # август хуже июля: чаще bad в оформлении
                zone1 = "bad" if (index % 4 == 0 if month == "08" else index % 10 == 0) else ("weak" if index % 5 == 0 else "ok")
                zone2a = "bad" if index % 8 == 0 else "ok"
                zone2b = "ok" if index % 3 else "na"
                kp = "matched" if index % 3 else ("unmatched" if index % 2 else "na")
                scored = index % 10 != 9  # каждый десятый без оценки (na)
                rows.append(
                    (
                        f"m{month}{index}",
                        str(int(month) * 1000 + index),
                        f"2026-{month}-{day:02d}",
                        "clinical_visit",
                        60.0 if scored else None,
                        doc_a if index % 2 else doc_b,
                        "Терапия" if index % 2 else "Кардиология",
                        "Центр" if index % 2 else "Юг",
                        "I10",
                        str(index),
                        "2026-09-30T00:00:00Z",
                        zone1 if scored else None,
                        zone2a if scored else None,
                        zone2b if scored else None,
                        kp if scored else None,
                        80.0 if scored else None,
                        70.0 if scored else None,
                        50.0 if scored else None,
                        "layer_v4" if scored else None,
                        "critical" if (month == "08" and index == 12) else None,
                    )
                )
        conn.executemany(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,doctor_key,specialty,filial,
                diagnosis_code,content_hash,updated_at,zone1_band,zone2a_band,zone2b_band,zone2b_kp_status,
                zone1_pct,zone2a_pct,zone2b_pct,layer_engine,overall_grade)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        findings = []
        for index in range(40):
            if index % 4 == 0:
                findings.append((f"m08{index}", "A_complaints_missing", "P2", 0, "Жалобы не описаны"))
            if index % 8 == 0:
                findings.append((f"m08{index}", "B_dx_not_justified", "P1", 0, "Диагноз не обоснован"))
            if index % 10 == 0:
                findings.append((f"m08{index}", "X_passed", "P3", 1, "Пройдено"))
        conn.executemany(
            "INSERT INTO fact_mo_finding(mis_id,finding_code,severity,passed,title_ru) VALUES(?,?,?,?,?)",
            findings,
        )
        conn.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "overview.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


BASE = {"period": "custom", "date_from": "2026-08-01", "date_to": "2026-08-31", "compare_period": "previous"}


def test_endpoint_declares_period_compare_and_facets() -> None:
    names = set(inspect.signature(api_methodist_mo_overview_dashboard).parameters)
    for key in ("period", "compare_period", "date_from", "date_to", "granularity",
                "specializations", "filials", "doctors", "document_kinds"):
        assert key in names


def test_blocks_present_and_consistent(warehouse: Path) -> None:
    out = mo_backend.build_overview_dashboard(dict(BASE))
    assert out["ok"] and out["available"]
    assert out["granularity"] == "day"
    # O1: лента по дням, сумма корзин = все случаи окна
    buckets = out["grades"]["buckets"]
    assert len(buckets) == 20
    assert sum(b["n"] for b in buckets) == 40 == out["grades"]["n"]
    assert out["grades"]["totals"]["na"] == 4  # каждый десятый без оценки
    assert out["grades"]["totals"]["critical"] == 1  # записанная скорером оценка первична
    assert set(out["grades"]["totals"]) == set(mo_backend.OVERVIEW_GRADE_KEYS)
    # O2: зоны только по оценённым, с дельтой к прошлому периоду
    zones = out["zones"]
    assert zones["zone1"]["n"] == 36 == out["coverage"]["evaluated"]
    assert zones["zone1"]["ok_pct"] is not None and zones["zone1"]["prev_ok_pct"] is not None
    assert zones["zone1"]["delta_ok_pct"] < 0  # август хуже июля по оформлению
    assert set(zones["zone1"]["bands"]) == {"ok", "weak", "bad", "na"}
    # O3: тренд текущего и прошлого периода
    # дни 10 и 20 целиком без оценки - в тренде зон их нет, в ленте оценок они есть как na
    assert len(out["trends"]) == 18 and len(out["trends_compare"]) == 18
    assert {b["date"] for b in buckets} - {t["date"] for t in out["trends"]} == {"2026-08-10", "2026-08-20"}
    assert {"date", "n_evaluated", "zone1_avg", "zone1_bad_pct"} <= set(out["trends"][0])
    # O4: тепловая карта - 2 специальности, недели окна, доля плохо
    heat = out["heatmap"]
    assert [r["specialty"] for r in heat["rows"]] == ["Кардиология", "Терапия"] or \
        sorted(r["specialty"] for r in heat["rows"]) == ["Кардиология", "Терапия"]
    assert heat["weeks"] and all(len(r["cells"]) == len(heat["weeks"]) for r in heat["rows"])
    assert any(c["bad_pct"] is not None for r in heat["rows"] for c in r["cells"])
    # O5: топ причин - только непройденные, отсортированы по числу случаев, с русской подписью
    top = out["findings_top"]
    assert [f["code"] for f in top] == ["A_complaints_missing", "B_dx_not_justified"]
    assert top[0]["n_cases"] == 10 and top[1]["n_cases"] == 5
    assert top[0]["label"] and top[0]["severity"] == "P2"
    # O6: воронка КП по тем же корзинам
    kp = out["kp_funnel"]
    assert sum(b["n"] for b in kp["buckets"]) == 40
    assert kp["totals"]["matched"] + kp["totals"]["unmatched"] + kp["totals"]["na"] == 40
    assert kp["totals"]["na"] >= 4  # без оценки -> na


def test_facets_apply_to_every_block(warehouse: Path) -> None:
    full = mo_backend.build_overview_dashboard(dict(BASE))
    part = mo_backend.build_overview_dashboard({**BASE, "specializations": "Терапия"})
    assert part["grades"]["n"] == 20 < full["grades"]["n"]
    assert part["zones"]["zone1"]["n"] < full["zones"]["zone1"]["n"]
    assert [r["specialty"] for r in part["heatmap"]["rows"]] == ["Терапия"]
    assert sum(f["n_cases"] for f in part["findings_top"]) < sum(f["n_cases"] for f in full["findings_top"])
    assert part["kp_funnel"]["totals"]["n"] == 20
    by_doctor = mo_backend.build_overview_dashboard({**BASE, "doctors": "Врач Б"})
    assert by_doctor["grades"]["n"] == 20
    by_filial = mo_backend.build_overview_dashboard({**BASE, "filials": "Юг"})
    assert by_filial["grades"]["n"] == 20


def test_granularity_follows_window_and_compare_none(warehouse: Path) -> None:
    ytd = mo_backend.build_overview_dashboard(
        {"period": "custom", "date_from": "2026-01-01", "date_to": "2026-08-31", "compare_period": "none"}
    )
    assert ytd["granularity"] == "month"
    assert [b["date"] for b in ytd["grades"]["buckets"]] == ["2026-07", "2026-08"]
    assert ytd["trends_compare"] == []
    assert ytd["zones"]["zone1"]["delta_ok_pct"] is None
    week = mo_backend.build_overview_dashboard({**BASE, "granularity": "week"})
    assert week["granularity"] == "week"
    assert all(b["date"].startswith("2026-W") for b in week["grades"]["buckets"])


def test_bad_granularity_is_422(warehouse: Path) -> None:
    with pytest.raises(ValueError):
        mo_backend.build_overview_dashboard({**BASE, "granularity": "hour"})


def test_small_heatmap_cells_are_marked_suppressed(warehouse: Path) -> None:
    out = mo_backend.build_overview_dashboard({**BASE, "doctors": "Врач А"})
    cells = [c for r in out["heatmap"]["rows"] for c in r["cells"] if 0 < c["n"] < mo_backend.SUPPRESSION_N]
    assert cells and all(c["suppressed"] for c in cells)
    assert out["heatmap"]["suppression_n"] == mo_backend.SUPPRESSION_N


def test_na_drills_from_overview_charts_filter_cases(warehouse: Path) -> None:
    """Клик по «Нет оценки» и «Без сравнения» на Обзоре должен резать /cases, а не сбрасывать фильтр."""
    from clinical_knowledge.mo_overall_grade import overall_grade_id

    dash = mo_backend.build_overview_dashboard(dict(BASE))
    na_total = dash["grades"]["totals"]["na"]
    kp_na_total = dash["kp_funnel"]["totals"]["na"]
    assert na_total > 0 and kp_na_total > 0

    cases_na = mo_backend.build_cases({**BASE, "overall_grade": "na", "document_kinds": "clinical_visit", "page_size": 200})
    assert cases_na["total"] == na_total
    assert cases_na["rows"]
    assert all(overall_grade_id(item) == "na" for item in cases_na["rows"])

    cases_kp_na = mo_backend.build_cases({**BASE, "kp_status": "na", "document_kinds": "clinical_visit", "page_size": 200})
    assert cases_kp_na["total"] == kp_na_total
    assert cases_kp_na["rows"]
    assert all(str(item.get("zone2b_kp_status") or "").lower() not in {"matched", "unmatched"} for item in cases_kp_na["rows"])

    cases_all = mo_backend.build_cases({**BASE, "document_kinds": "clinical_visit", "page_size": 200})
    assert cases_all["total"] > cases_kp_na["total"]
    assert cases_all["total"] > cases_na["total"]
