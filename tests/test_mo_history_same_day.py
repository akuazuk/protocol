"""Synthetic reproduction: same-day history cutoff and query failure."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import initialize_warehouse, patient_key_for
from clinical_knowledge.mo_history_assessment import build_history_assessment_context
from clinical_knowledge.mo_patient_history_bundle import build_patient_history_bundle


def _seed_same_day(warehouse: Path) -> None:
    initialize_warehouse(warehouse)
    pk = patient_key_for("3003")
    with sqlite3.connect(warehouse) as db:
        cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
        if "visit_at" not in cols:
            db.execute("ALTER TABLE fact_mo_case ADD COLUMN visit_at TEXT")
        rows = (
            ("s1", "v-prev-day", "2026-03-11", "2026-03-11T18:00:00+03:00"),
            ("s2", "v-morning", "2026-03-12", "2026-03-12T09:10:00+03:00"),
            ("s3", "v-current", "2026-03-12", "2026-03-12T14:00:00+03:00"),
            ("s4", "v-evening", "2026-03-12", "2026-03-12T16:40:00+03:00"),
            ("s5", "v-unknown", "2026-03-12", None),
        )
        for mis, vid, day, visit_at in rows:
            db.execute(
                """INSERT INTO fact_mo_case
                   (mis_id, visit_id, visit_date, visit_at, document_kind, overall_pct,
                    status, doctor_key, doctor_id, specialty, patient_key,
                    diagnosis_code, content_hash, updated_at)
                   VALUES (?, ?, ?, ?, 'clinical_visit', 70, 'review',
                    'dk_a', '11', 'Уролог', ?, 'N30.0', 'h', 'now')""",
                (mis, vid, day, visit_at, pk),
            )
        db.commit()


def test_same_day_earlier_included_later_and_unknown_excluded(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_same_day(warehouse)
    bundle = build_patient_history_bundle(
        patient_id="3003",
        as_of_date="2026-03-12",
        cutoff_at="2026-03-12T14:00:00+03:00",
        doctor_id="11",
        doctor_key="dk_a",
        specialty="Уролог",
        current_code="N30.0",
        exclude_ids={"s3", "v-current"},
        warehouse=warehouse,
    )
    included = {
        row["visit_id"]
        for row in (*bundle["same_doctor"], *bundle["same_specialty"], *bundle["other"])
    }
    assert "v-prev-day" in included
    assert "v-morning" in included
    assert "v-current" not in included
    assert "v-evening" not in included
    assert "v-unknown" not in included
    excluded = {row["visit_id"]: row["exclusion_reason"] for row in bundle["excluded_visits"]}
    assert excluded["v-evening"] == "same_day_after_cutoff"
    assert excluded["v-unknown"] == "unknown_time"
    assert bundle["ok"] is True
    assert bundle["status"] == "has_priors"
    assert bundle["cutoff_at"] == "2026-03-12T14:00:00+03:00"


def test_unknown_same_day_does_not_grant_correction(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_same_day(warehouse)
    bundle = build_patient_history_bundle(
        patient_id="3003",
        as_of_date="2026-03-12",
        cutoff_at="2026-03-12",
        doctor_id="11",
        doctor_key="dk_a",
        specialty="Уролог",
        current_code="N30.0",
        exclude_ids={"s3", "v-current"},
        warehouse=warehouse,
    )
    included = {
        row["visit_id"]
        for row in (*bundle["same_doctor"], *bundle["same_specialty"], *bundle["other"])
    }
    assert "v-prev-day" in included
    assert "v-morning" not in included
    assert "v-evening" not in included
    context = build_history_assessment_context(
        history_bundle=bundle,
        current_code="N30.0",
        cutoff_at="2026-03-12",
    )
    assert context["any_prior_exists"] is True
    assert context["correction_assessable"] is False


def test_query_failed_is_not_available_history() -> None:
    failed = {
        "ok": False,
        "reason": "query_failed",
        "status": "error",
        "summary": {"n_visits": 0},
    }
    context = build_history_assessment_context(history_bundle=failed)
    assert context["history_available"] is False
    assert context["status"] == "error"
    assert "history_unavailable" in context["exclusion_reason_codes"]
