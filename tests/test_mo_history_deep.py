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
