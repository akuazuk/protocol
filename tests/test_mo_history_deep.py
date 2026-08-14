"""Слой B: shadow-кредит эпизода без чтения CSV."""
from __future__ import annotations

from clinical_knowledge.mo_history_continuity import MODE_KNOWN_DOCTOR
from clinical_knowledge.mo_history_deep import public_deep_for_ui, shadow_history_credit_finding


def test_shadow_credit_only_for_known_episode():
    finding = shadow_history_credit_finding(
        {
            "continuity": {
                "known_episode": True,
                "mode": MODE_KNOWN_DOCTOR,
                "mode_ru": "Продолжение случая у этого врача",
                "last_matched_date": "2026-07-20",
            },
            "already_slots": ["anamnesis_doctor", "clinical_diagnosis", "treatment_recommendations"],
            "prior_visit_date": "2026-07-20",
        }
    )
    assert finding is not None
    assert finding["shadow"] is True
    assert finding["code"] == "B_history_episode_credit"
    assert "анамнез" in finding["detail_ru"]
    assert "не меняем" in finding["detail_ru"].lower() or "не меняем" in finding["detail_ru"]


def test_no_credit_without_episode():
    assert shadow_history_credit_finding({"continuity": {"known_episode": False}}) is None


def test_public_deep_strips_clinical_text():
    pub = public_deep_for_ui(
        {
            "prior_n_loaded": 1,
            "prior_visit_date": "2026-07-20",
            "already_slots": ["complaints"],
            "prior_slots": [{"visit_date": "2026-07-20", "present_slots": ["complaints"]}],
            "prior_clinical": {"complaints": "secret"},
        }
    )
    assert "prior_clinical" not in pub
    assert pub["already_slots"] == ["complaints"]
