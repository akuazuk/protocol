"""Слой B: shadow-кредит эпизода без чтения CSV."""
from __future__ import annotations

import os

from clinical_knowledge.mo_history_continuity import MODE_KNOWN_DOCTOR
from clinical_knowledge.mo_history_deep import public_deep_for_ui, shadow_history_credit_finding
from scripts.run_mo_history_deep import _sanitize_llm_error, gemini_key_env_names


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


def test_gemini_keys_prefer_billed_and_dedupe(monkeypatch) -> None:
    monkeypatch.setenv("GENERATIVE_LANGUAGE_API_KEY", "AQ.billed")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-studio")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-studio")
    monkeypatch.setenv("GOOGLE_API_KEY_2", "AIza-studio-2")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    assert gemini_key_env_names() == [
        "GENERATIVE_LANGUAGE_API_KEY",
        "GOOGLE_API_KEY_2",
        "GOOGLE_API_KEY",
    ]


def test_sanitize_llm_error_strips_key(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-secret-value")
    text = _sanitize_llm_error(RuntimeError("429 cap AIza-secret-value"))
    assert "AIza-secret-value" not in text
    assert "[key]" in text


def test_prior_selection_excludes_unrelated_rich_record(monkeypatch):
    from clinical_knowledge import mo_history_deep as deep

    bundle = {"same_doctor": [
        {"visit_id": "synthetic-other", "visit_date": "2026-08-19", "diagnosis_code": "K29"},
        {"visit_id": "synthetic-match", "visit_date": "2026-08-18", "diagnosis_code": "J06"},
    ]}
    loaded = []

    def load(visits, *, limit):
        loaded.extend(visits[:limit])
        return [{"visit_date": v["visit_date"], "present_slots": ["clinical_diagnosis"], "clinical": {"marker": v["visit_id"]}} for v in visits[:limit]]

    monkeypatch.setattr(deep, "load_prior_slots_for_visits", load)
    result = deep.pick_episode_prior(history_bundle=bundle, current_code="J06.9", limit=1)
    assert [v["visit_id"] for v in loaded] == ["synthetic-match"]
    assert result["prior_clinical"]["marker"] == "synthetic-match"


def test_prior_recency_precedes_completeness(monkeypatch):
    from clinical_knowledge import mo_history_deep as deep

    bundle = {"same_specialty": [
        {"visit_id": "synthetic-old", "visit_date": "2026-08-01", "diagnosis_code": "J06"},
        {"visit_id": "synthetic-new", "visit_date": "2026-08-18", "diagnosis_code": "J06"},
    ]}

    def load(visits, *, limit):
        assert visits[0]["visit_id"] == "synthetic-new"
        return [{"visit_date": v["visit_date"], "present_slots": ["clinical_diagnosis"] * (5 if v["visit_id"] == "synthetic-old" else 1), "clinical": {"marker": v["visit_id"]}} for v in visits[:limit]]

    monkeypatch.setattr(deep, "load_prior_slots_for_visits", load)
    result = deep.pick_episode_prior(history_bundle=bundle, current_code="J06.9")
    assert result["prior_clinical"]["marker"] == "synthetic-new"
    assert result["prior_n_loaded"] == 2


def test_unmatched_episode_does_not_load_clinical_slots(monkeypatch):
    from clinical_knowledge import mo_history_deep as deep

    def load(visits, *, limit):
        assert visits == []
        return []

    monkeypatch.setattr(deep, "load_prior_slots_for_visits", load)
    result = deep.pick_episode_prior(history_bundle={"same_doctor": [{"visit_id": "synthetic-other", "diagnosis_code": "K29"}]}, current_code="J06.9")
    assert result["prior_clinical"] is None
    assert result["prior_n_loaded"] == 0
