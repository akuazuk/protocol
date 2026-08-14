"""Непрерывность эпизода и сортировка на глубокий прогон."""
from __future__ import annotations

from clinical_knowledge.mo_history_continuity import (
    MODE_KNOWN_DOCTOR,
    MODE_NEW_DOCTOR,
    MODE_NONE,
    TRACK_HISTORY,
    TRACK_SAFETY,
    TRACK_STRONG,
    evaluate_history_continuity,
    rank_for_deep_run,
)


def _bundle(*, doctor=None, specialty=None):
    return {
        "same_doctor": doctor or [],
        "same_specialty": specialty or [],
        "other": [],
        "summary": {
            "n_same_doctor": len(doctor or []),
            "n_same_specialty": len(specialty or []),
            "n_visits": len(doctor or []) + len(specialty or []),
            "current_code": "N30.0",
        },
        "tier": "known_to_doctor" if doctor else "first_contact",
    }


def test_known_episode_same_doctor_and_history_track():
    out = evaluate_history_continuity(
        current_code="N30.0",
        current_text="острый цистит",
        history_bundle=_bundle(
            doctor=[{"visit_date": "2026-07-20", "diagnosis_code": "N30.0", "diagnosis_text": "цистит"}]
        ),
        zones={"zone2a_band": "bad", "zone2b_band": "ok"},
    )
    assert out["mode"] == MODE_KNOWN_DOCTOR
    assert out["known_episode"] is True
    assert "diagnosis" in out["already_described"]
    assert out["deep_run_track"] == TRACK_HISTORY
    assert out["deep_run_score"] == 150


def test_new_problem_on_known_doctor():
    out = evaluate_history_continuity(
        current_code="J06.9",
        current_text="орви",
        history_bundle=_bundle(
            doctor=[{"visit_date": "2026-07-20", "diagnosis_code": "N30.0", "diagnosis_text": "цистит"}]
        ),
        zones={"zone2a_band": "weak", "zone2b_band": "bad"},
    )
    assert out["mode"] == MODE_NEW_DOCTOR
    assert out["known_episode"] is False
    assert out["deep_run_track"] == TRACK_HISTORY
    assert out["deep_run_score"] == 80


def test_no_history_poor_dx_goes_to_strong_model():
    out = evaluate_history_continuity(
        current_code="G43.1",
        history_bundle={"same_doctor": [], "same_specialty": [], "other": [], "summary": {"n_visits": 0}},
        zones={"zone2a_band": "bad"},
        history_tier="first_contact",
        history_prior_n=0,
    )
    assert out["mode"] == MODE_NONE
    assert out["deep_run_track"] == TRACK_STRONG
    assert out["deep_run_score"] == 40


def test_safety_outranks_history():
    out = evaluate_history_continuity(
        current_code="N30.0",
        history_bundle=_bundle(
            doctor=[{"visit_date": "2026-07-20", "diagnosis_code": "N30.0"}]
        ),
        zones={"zone2a_band": "bad"},
        attention_primary="safety",
    )
    assert out["deep_run_track"] == TRACK_SAFETY
    assert out["deep_run_score"] == 200


def test_queue_fallback_uses_warehouse_tier():
    out = evaluate_history_continuity(
        current_code="I21.0",
        zones={"zone2a_band": "bad"},
        history_tier="known_to_doctor",
        history_prior_n=3,
    )
    assert out["mode"] == MODE_KNOWN_DOCTOR
    assert out["deep_run_track"] == TRACK_HISTORY


def test_rank_orders_history_before_lonely_poor():
    items = [
        {"case_id": "a", "deep_run_score": 40, "overall_pct": 40},
        {"case_id": "b", "deep_run_score": 150, "overall_pct": 55},
        {"case_id": "c", "deep_run_score": 200, "overall_pct": 70},
    ]
    ordered = sorted(items, key=rank_for_deep_run)
    assert [row["case_id"] for row in ordered] == ["c", "b", "a"]
