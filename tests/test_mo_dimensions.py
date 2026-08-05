from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse

ROOT = Path(__file__).resolve().parents[1]


def _seed(path: Path) -> tuple[str, str]:
    initialize_warehouse(path)
    doctor_a = doctor_key_for("Врач А")
    doctor_b = doctor_key_for("Врач Б")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [
                (doctor_a, "Врач А", "Терапия", "Центр"),
                (doctor_b, "Врач Б", "Терапия", "Центр"),
            ],
        )
        for index in range(50):
            doctor_key = doctor_a if index < 25 else doctor_b
            score = 60.0 if index < 25 else 80.0
            mis_id = f"case-{index}"
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                    doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    f"visit-{index}",
                    f"2026-07-{20 + index % 5:02d}",
                    "consultation",
                    score,
                    "review",
                    doctor_key,
                    "Терапия",
                    "Центр",
                    "I10" if index % 2 else "J06",
                    "Болезни системы кровообращения" if index % 2 else "Болезни органов дыхания",
                    str(index),
                    "2026-07-30T00:00:00Z",
                ),
            )
            conn.execute(
                """INSERT INTO fact_mo_finding
                   (mis_id,finding_code,severity,passed,evidence,source_ref)
                   VALUES(?,?,?,?,?,?)""",
                (
                    mis_id,
                    "S_red_flag" if index < 3 else "B_gap",
                    "P0" if index < 3 else "P2",
                    0,
                    "",
                    "protocol:55:1",
                ),
            )
        conn.commit()
    return doctor_a, doctor_b


def _params() -> dict[str, str]:
    return {"period": "custom", "date_from": "2026-07-20", "date_to": "2026-07-24"}


def test_dimension_sql_contracts_n_gate_ci_and_no_raw_ranking(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor_a, _ = _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")

    doctors = mo_backend.build_dimension("doctors", _params())
    assert doctors["source"] == "warehouse"
    assert doctors["sample_gate"] == 20
    assert doctors["ranking_metric"] == "expected_delta"
    assert doctors["no_raw_score_ranking"] is True
    assert [item["delta"] for item in doctors["ranking"]] == sorted(
        item["delta"] for item in doctors["ranking"]
    )
    selected = next(item for item in doctors["items"] if item["key"] == doctor_a)
    assert selected["n"] == 25
    # n-gate открывает карточку; ranking требует case_mix_reliable (R² >= 0.30)
    assert selected["enough_data"] is True
    assert selected["case_mix_reliable"] is False
    assert selected["case_mix_model"]["valid"] is False
    assert selected["case_mix_model"]["r_squared"] < 0.3
    assert selected["key"] not in {item["key"] for item in doctors["ranking"]}
    assert selected["delta_ci95"]["low"] is not None
    assert selected["p0_cases"] == 3

    specialties = mo_backend.build_dimension("specialties", _params())
    assert specialties["items"][0]["boxplot"] == [60.0, 60.0, 70.0, 80.0, 80.0]
    diagnoses = mo_backend.build_dimension("diagnoses", _params())
    assert diagnoses["encoding"] == {"size": "volume", "color": "avg_score"}
    assert all(item["value"] >= mo_backend.SUPPRESSION_N for item in diagnoses["items"])
    safety = mo_backend.build_dimension("safety", _params())
    assert safety["incidents"][0]["source_ref"] == "protocol:55:1"


def test_specialty_doctor_case_finding_source_drilldown(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor_a, _ = _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")

    doctors = mo_backend.build_drilldown("specialty", "Терапия", _params())["items"]
    assert doctor_a in {item["id"] for item in doctors}
    cases = mo_backend.build_drilldown("doctor", doctor_a, _params())["items"]
    assert cases and cases[0]["level"] == "case"
    findings = mo_backend.build_drilldown("case", cases[0]["mis_id"], _params())["items"]
    assert findings and findings[0]["source_ref"] == "protocol:55:1"


def test_phase5_frontend_uses_real_echarts_and_selected_doctor_action_flow() -> None:
    html = (ROOT / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
    for marker in (
        "doctor-scatter-chart",
        "specialty-boxplot-chart",
        "icd-treemap-chart",
        "safety-severity-chart",
        "doctor-cabinet-records",
        "doctor-template-pairs",
        "access-log-content",
    ):
        assert marker in html
    assert 'type:"scatter"' in script
    assert 'type:"boxplot"' in script
    assert 'type:"treemap"' in script
    assert 'stack:"severity"' in script
    assert "brushSelected" in script
    assert "open-selected-doctors" in script
    assert "/doctor-cabinet/disputes" in script


def test_dimension_routes_require_authentication(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _seed(db)
    monkeypatch.setenv("METHODIST_TOKEN", "test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    client = TestClient(rag_server.app)
    path = (
        "/api/methodist/mo/dimensions/doctors"
        "?period=custom&date_from=2026-07-20&date_to=2026-07-24"
    )
    assert client.get(path).status_code == 403
    response = client.get(path, headers={"X-Methodist-Token": "test-token"})
    assert response.status_code == 200
    assert response.json()["ranking_metric"] == "expected_delta"
