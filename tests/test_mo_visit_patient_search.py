"""Поиск случаев МО по visit_id / patient_id."""
from __future__ import annotations

from clinical_knowledge.mis_kz_quality import _match_filters
from clinical_knowledge.mo_backend import _filter_records, _identity_lookup
from clinical_knowledge.mo_daily import patient_key_for


def test_identity_lookup_from_q_digits() -> None:
    assert _identity_lookup({"q": "3468853"}) == {"q_id": "3468853"}
    assert _identity_lookup({"visit_id": "3468853", "q": "x"}) == {"visit_id": "3468853"}
    assert _identity_lookup({"patient_id": "532264"}) == {"patient_id": "532264"}
    assert _identity_lookup({"q": "abc"}) == {}


def test_match_filters_by_visit_and_patient_id() -> None:
    rec = {
        "visit_id": "3468853",
        "case_id": "3468853",
        "mis_id": "860796",
        "patient_id": "532264",
        "patient_key": patient_key_for("532264"),
        "doctor_fio": "Русаков",
        "diagnosis_short": "Миозит",
        "mkb_code_main": "M60",
        "date": "2026-06-10",
    }
    assert _match_filters(rec, {"visit_id": "3468853"})
    assert not _match_filters(rec, {"visit_id": "999"})
    assert _match_filters(rec, {"patient_id": "532264"})
    assert not _match_filters(rec, {"patient_id": "1"})
    assert _match_filters(rec, {"q": "3468853"})
    assert _match_filters(rec, {"q": "532264"})
    assert _match_filters(rec, {"q": "миозит"})


def test_filter_records_id_lookup_ignores_date_window() -> None:
    rows = [
        {
            "case_id": "3468853",
            "visit_id": "3468853",
            "mis_id": "860796",
            "patient_id": "532264",
            "patient_key": patient_key_for("532264"),
            "date": "2026-06-10",
            "doctor_fio": "Русаков",
            "diagnosis_short": "Миозит",
            "mkb_code_main": "M60",
            "document_kind": "clinical_visit",
            "specialization": "Ортопед-травматолог",
            "filial": "Захарова",
            "status": "good",
            "overall_pct": 83.5,
            "p0": 0,
            "p1": 1,
        }
    ]
    out = _filter_records(
        rows,
        {
            "date_from": "2026-08-09",
            "date_to": "2026-08-09",
            "visit_id": "3468853",
            "document_kinds": "clinical_visit",
        },
    )
    assert len(out) == 1
    out_q = _filter_records(
        rows,
        {
            "date_from": "2026-08-09",
            "date_to": "2026-08-09",
            "q": "532264",
            "document_kinds": "clinical_visit",
        },
    )
    assert len(out_q) == 1
