from __future__ import annotations

from clinical_knowledge.mo_overall_grade import compute_mo_overall_grade
from clinical_knowledge.mo_zone_scores import compute_mo_zone_scores


def test_safety_critical_is_kritichno() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "ok",
            "zone2a_band": "ok",
            "zone2b_band": "ok",
            "zone2b_kp_status": "matched",
            "safety": {"band": "critical", "codes": ["C_red_flag_unrouted"]},
        }
    )
    assert out["grade"] == "critical"
    assert out["label_ru"] == "Критично"


def test_safety_important_is_vazhno() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "weak",
            "zone2a_band": "ok",
            "zone2b_band": "na",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "important", "codes": ["C_nsaid_dup"]},
        }
    )
    assert out["grade"] == "important"
    assert out["label_ru"] == "Важно"


def test_dx_bad_is_vazhno() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "ok",
            "zone2a_band": "bad",
            "zone2b_band": "na",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "none"},
        }
    )
    assert out["grade"] == "important"


def test_plan_bad_only_if_kp_matched() -> None:
    matched = compute_mo_overall_grade(
        {
            "zone1_band": "ok",
            "zone2a_band": "ok",
            "zone2b_band": "bad",
            "zone2b_kp_status": "matched",
            "safety": {"band": "none"},
        }
    )
    assert matched["grade"] == "poor"
    unmatched = compute_mo_overall_grade(
        {
            "zone1_band": "ok",
            "zone2a_band": "ok",
            "zone2b_band": "bad",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "none"},
        }
    )
    assert unmatched["grade"] == "fair"


def test_zone1_bad_is_slabo() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "bad",
            "zone2a_band": "ok",
            "zone2b_band": "na",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "none"},
        }
    )
    assert out["grade"] == "poor"
    assert out["label_ru"] == "Слабо"


def test_zone1_weak_is_fair() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "weak",
            "zone2a_band": "ok",
            "zone2b_band": "na",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "none"},
        }
    )
    assert out["grade"] == "fair"
    assert out["label_ru"] == "С замечанием"


def test_all_ok_is_good() -> None:
    out = compute_mo_overall_grade(
        {
            "zone1_band": "ok",
            "zone2a_band": "ok",
            "zone2b_band": "na",
            "zone2b_kp_status": "unmatched",
            "safety": {"band": "none"},
        }
    )
    assert out["grade"] == "good"
    assert out["label_ru"] == "Хорошо"


def test_rceth_contra_lifts_only_when_primary() -> None:
    base = {
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "ok",
        "zone2b_kp_status": "matched",
        "safety": {"band": "none"},
        "rceth_codes": ["C_rceth_contraindication"],
    }
    shadow = compute_mo_overall_grade(base, rceth_primary=False)
    assert shadow["grade"] == "good"
    primary = compute_mo_overall_grade(base, rceth_primary=True)
    assert primary["grade"] == "important"


def test_live_zone_engine_thin_orvi_is_not_good() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": {"clinical_diagnosis": "ОРВИ", "exam_recommendations": "ОАК"},
            "meta": {"visit_date": "2026-08-02", "visit_time": "09:00"},
            "findings": [],
            "document_kind": "clinical_visit",
        }
    )
    grade = compute_mo_overall_grade(zones)
    assert zones["zone1_band"] == "bad"
    # Пустые жалобы/осмотр роняют и опору диагноза → Важно, не «Слабо».
    assert grade["grade"] in {"important", "poor"}
    assert grade["grade"] != "good"
