"""E3: labels, warehouse shadow findings, linked fields."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from clinical_knowledge.mo_concordance_findings import (
    evaluate_mo_concordance,
    merge_concordance_into_findings,
)
from clinical_knowledge.mo_daily import add_document_taxonomy, build_daily_report, upsert_warehouse
from clinical_knowledge.mo_finding_labels_ru import FINDING_TITLE_RU, finding_label_ru
from tests.test_mo_concordance_smirnova import SMIRNOVA_CASE
from tests.test_mo_daily_pipeline import wide_frame


def test_concordance_labels_are_russian() -> None:
    for code in (
        "finding_not_in_diagnosis",
        "underworkup_chronic_red_flag",
        "anamnesis_thin_for_duration",
        "plan_laterality_mismatch",
        "icd_weakly_supported",
        "pediatric_limp_ddx_not_addressed",
    ):
        assert code in FINDING_TITLE_RU
        label = finding_label_ru(code, code)
        assert label != code
        assert any(ch.isalpha() and ord(ch) > 127 for ch in label)


def test_finding_not_in_diagnosis_has_linked_fields() -> None:
    findings = evaluate_mo_concordance(SMIRNOVA_CASE)
    target = next(f for f in findings if f["code"] == "finding_not_in_diagnosis")
    assert "objective_status" in target["linked_fields"]
    assert "clinical_diagnosis" in target["linked_fields"]
    assert target["link_hint_ru"]


def test_merge_concordance_adds_shadow_without_dupes() -> None:
    primary = [{"code": "C_red_flag", "severity": "P1", "passed": False, "title_ru": "Красный флаг"}]
    merged = merge_concordance_into_findings(primary, SMIRNOVA_CASE)
    codes = [f["code"] for f in merged]
    assert "C_red_flag" in codes
    assert "finding_not_in_diagnosis" in codes
    again = merge_concordance_into_findings(merged, SMIRNOVA_CASE)
    assert codes.count("finding_not_in_diagnosis") == 1
    assert [f["code"] for f in again].count("finding_not_in_diagnosis") == 1


def test_warehouse_writes_shadow_p1_for_yesterday_queue(tmp_path: Path, monkeypatch) -> None:
    day = "2026-08-04"
    frame = wide_frame(1, day=day)
    frame.loc[0, "mkb_codes"] = "M60"
    frame["service_codes"] = "10.100.1."
    frame["service_names"] = "Консультация"
    raw = add_document_taxonomy(frame).to_dict(orient="records")
    shadow = evaluate_mo_concordance(SMIRNOVA_CASE)
    assert any(f["code"] == "finding_not_in_diagnosis" and f["severity"] == "P1" for f in shadow)
    cases = [
        {
            "mis_id": raw[0]["id"],
            "visit_id": raw[0]["visit_id"],
            "doctor_fio": "Тестов Т.Т.",
            "doctor_specialization": "Ортопед",
            "filial": "A",
            "overall_pct": 88,
            "status": "mostly_compliant",
            "deep": {
                "axes": {"documentation": 80, "clinical_concordance": 70, "safety": 90},
                "findings": [],
                "n_by_severity": {},
                "shadow_findings": shadow,
            },
        }
    ]
    secure, _public = build_daily_report(
        raw, cases, day=date.fromisoformat(day), run_id="e3", revision=1, quality={"passed": True}
    )
    warehouse = tmp_path / "warehouse.sqlite"
    written = upsert_warehouse(warehouse, raw, cases, secure)
    assert written.get("fact_mo_finding_shadow", 0) >= 1

    with sqlite3.connect(warehouse) as db:
        row = db.execute(
            """SELECT finding_code, severity, is_shadow, linked_fields_json
               FROM fact_mo_finding WHERE finding_code=?""",
            ("finding_not_in_diagnosis",),
        ).fetchone()
        assert row is not None
        assert row[1] == "P1"
        assert int(row[2]) == 1
        assert "objective_status" in (row[3] or "")

    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    from clinical_knowledge import mo_backend

    report = mo_backend.build_daily_report(day)
    items = (report.get("action_cases") or {}).get("items") or []
    codes = {item.get("finding_code") for item in items}
    # Shadow concordance остаётся в warehouse, но не создаёт тикет очереди разбора.
    assert "finding_not_in_diagnosis" not in codes
