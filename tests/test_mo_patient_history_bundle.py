"""Patient history bundle A1-A4: shelves, tiers, one finding."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.mo_daily import initialize_warehouse, patient_key_for
from clinical_knowledge.mo_patient_history_bundle import (
    FINDING_CODE,
    TIER_FIRST_CONTACT,
    TIER_KNOWN_DOCTOR,
    TIER_KNOWN_SPECIALTY,
    attach_bundle_to_case,
    build_patient_history_bundle,
    evaluate_history_mo,
    merge_patient_history_into_findings,
    name_match_threshold_delta,
    public_bundle_for_ui,
)


def _seed(warehouse: Path) -> None:
    initialize_warehouse(warehouse)
    import sqlite3

    pk = patient_key_for("1001")
    rows = [
        ("1", "v1", "2026-01-10", "dk_a", "11", "Уролог", "N30.0", "цистит", 80.0),
        ("2", "v2", "2026-02-10", "dk_b", "22", "Уролог", "N30.0", "цистит", 78.0),
        ("3", "v3", "2026-03-12", "dk_a", "11", "Уролог", "N30.0", "цистит", 82.0),
        ("4", "v4", "2026-05-01", "dk_a", "11", "Уролог", "N39.0", "другое", 71.0),
        ("5", "v5", "2026-04-01", "dk_c", "33", "Терапевт", "J06.9", "орви", 90.0),
    ]
    with sqlite3.connect(warehouse) as db:
        for mis, vid, day, dkey, did, spec, code, text, pct in rows:
            db.execute(
                """INSERT INTO fact_mo_case
                   (mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                    doctor_key, doctor_id, specialty, patient_key, diagnosis_code,
                    diagnosis_text, content_hash, updated_at)
                   VALUES (?, ?, ?, 'medical_exam', ?, 'review', ?, ?, ?, ?, ?, ?, 'h', 'now')""",
                (mis, vid, day, pct, dkey, did, spec, pk, code, text),
            )
        db.commit()


def test_patient_key_stable_hash() -> None:
    assert patient_key_for("42") == patient_key_for("42")
    assert patient_key_for("42") != patient_key_for("43")
    assert len(patient_key_for("42")) == 20


def test_bundle_shelves_and_excludes_current(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    bundle = build_patient_history_bundle(
        patient_id="1001",
        as_of_date="2026-08-01",
        doctor_id="11",
        doctor_key="dk_a",
        specialty="Уролог",
        current_code="N30.0",
        exclude_ids={"99"},
        warehouse=warehouse,
    )
    assert bundle["summary"]["n_same_doctor"] == 3
    assert bundle["summary"]["n_same_specialty"] == 1
    assert bundle["summary"]["n_other"] == 1
    assert bundle["tier"] == TIER_KNOWN_DOCTOR
    assert all(r["visit_date"] < "2026-08-01" for r in bundle["same_doctor"])


def test_current_visit_not_in_bundle(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    bundle = build_patient_history_bundle(
        patient_id="1001",
        as_of_date="2026-03-12",
        doctor_id="11",
        doctor_key="dk_a",
        specialty="Уролог",
        current_code="N30.0",
        exclude_ids={"3"},
        warehouse=warehouse,
    )
    assert all(r["mis_id"] != "3" for r in bundle["same_doctor"])
    assert bundle["summary"]["n_same_doctor"] == 1


def test_first_contact_and_specialty_only(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    first = build_patient_history_bundle(
        patient_id="1001",
        as_of_date="2026-08-01",
        doctor_id="99",
        doctor_key="dk_new",
        specialty="Уролог",
        current_code="N30.0",
        warehouse=warehouse,
    )
    assert first["tier"] == TIER_FIRST_CONTACT
    assert first["summary"]["n_same_doctor"] == 0
    assert first["summary"]["current_code_seen_in_specialty"] is True

    new_code = build_patient_history_bundle(
        patient_id="1001",
        as_of_date="2026-08-01",
        doctor_id="11",
        doctor_key="dk_a",
        specialty="Уролог",
        current_code="Z99.9",
        warehouse=warehouse,
    )
    assert new_code["tier"] == "new_for_profile"


def test_known_in_specialty_only_tier(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    # врач dk_a имел N39 и N30; коллега dk_b имел N30. Для кода только у коллеги -
    # возьмём врача без кода N30: нет такого в seed с визитами. Синтетика:
    import sqlite3

    pk = patient_key_for("2002")
    with sqlite3.connect(warehouse) as db:
        db.execute(
            """INSERT INTO fact_mo_case
               (mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                doctor_key, doctor_id, specialty, patient_key, diagnosis_code,
                content_hash, updated_at)
               VALUES ('10','v10','2026-01-01','medical_exam',70,'review',
                'dk_x','90','Уролог',?,'I10','h','now')""",
            (pk,),
        )
        db.execute(
            """INSERT INTO fact_mo_case
               (mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                doctor_key, doctor_id, specialty, patient_key, diagnosis_code,
                content_hash, updated_at)
               VALUES ('11','v11','2026-02-01','medical_exam',70,'review',
                'dk_y','91','Уролог',?,'N30.0','h','now')""",
            (pk,),
        )
        db.commit()
    bundle = build_patient_history_bundle(
        patient_id="2002",
        as_of_date="2026-08-01",
        doctor_id="90",
        doctor_key="dk_x",
        specialty="Уролог",
        current_code="N30.0",
        warehouse=warehouse,
    )
    assert bundle["tier"] == TIER_KNOWN_SPECIALTY


def test_insufficient_no_finding() -> None:
    bundle = build_patient_history_bundle(
        patient_id="",
        as_of_date="2026-08-01",
        doctor_key="x",
        specialty="Уролог",
    )
    assert evaluate_history_mo(bundle) == []


def test_one_history_finding(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    case = {
        "patient_id": "1001",
        "visit_date": "2026-08-01",
        "doctor_id": "11",
        "doctor_key": "dk_a",
        "specialty": "Уролог",
        "diagnosis_code": "N30.0",
        "mis_id": "99",
    }
    attach_bundle_to_case(case, warehouse=warehouse)
    findings = evaluate_history_mo(case["_patient_history"])
    assert len(findings) == 1
    assert findings[0]["code"] == FINDING_CODE
    assert findings[0]["history_tier"] == TIER_KNOWN_DOCTOR


def test_name_match_threshold_delta() -> None:
    assert name_match_threshold_delta({"current_code_seen_by_doctor": True}) < 0
    assert name_match_threshold_delta({"tier": "first_contact"}) > 0


def test_force_rebuild_after_stale_finding(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed(warehouse)
    case = {
        "patient_id": "1001",
        "visit_date": "2026-08-01",
        "doctor_id": "11",
        "doctor_key": "dk_a",
        "specialty": "Уролог",
        "diagnosis_code": "N30.0",
        "mis_id": "99",
    }
    stale = [
        {
            "code": FINDING_CODE,
            "history_tier": TIER_FIRST_CONTACT,
            "title_ru": "stale",
        }
    ]
    out = merge_patient_history_into_findings(stale, case, warehouse=warehouse, force=True)
    assert len([f for f in out if f.get("code") == FINDING_CODE]) == 1
    assert out[-1]["history_tier"] == TIER_KNOWN_DOCTOR
    assert int((case["_patient_history"]["summary"] or {}).get("n_visits") or 0) >= 3
    pub = public_bundle_for_ui(case["_patient_history"])
    assert pub["tier_label_ru"]
    assert "shadow" in pub["usage_for_scores_ru"].lower() or "не меняет" in pub["usage_for_scores_ru"]
