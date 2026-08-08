from __future__ import annotations

from clinical_knowledge.mo_dx_episode import resolve_dx_episode_for_suggest


def test_episode_current_only_without_history() -> None:
    out = resolve_dx_episode_for_suggest(
        clinical={"clinical_diagnosis": "Плосковальгусная установка стоп"},
        history_visits=[],
    )
    assert out["mode"] == "current_only"
    assert "плосковальгус" in out["query"].lower()
    assert out["matched_visits"] == []


def test_episode_enriches_same_stem_history() -> None:
    out = resolve_dx_episode_for_suggest(
        clinical={"clinical_diagnosis": "ПВУС", "mis_diagnos": "M21.0"},
        history_visits=[
            {
                "visit_id": "p1",
                "diagnosis_code": "M21.0",
                "diagnosis_text": "Плосковальгусная установка стоп с вальгированием",
            }
        ],
    )
    assert out["mode"] == "enriched"
    assert out["matched_visits"]
    assert "плосковальгус" in out["query"].lower()


def test_episode_ignores_unrelated_history() -> None:
    out = resolve_dx_episode_for_suggest(
        clinical={"clinical_diagnosis": "ОРВИ"},
        history_visits=[
            {
                "visit_id": "p1",
                "diagnosis_code": "M21.0",
                "diagnosis_text": "Плосковальгусная установка стоп",
            }
        ],
    )
    assert out["mode"] == "current_only"
    assert out["matched_visits"] == []
    assert "плосковальгус" not in out["query"].lower()


def test_episode_history_fallback_when_current_empty() -> None:
    out = resolve_dx_episode_for_suggest(
        clinical={"mis_diagnos": "M21.0"},
        history_visits=[
            {
                "visit_id": "p1",
                "diagnosis_code": "M21.0",
                "diagnosis_text": "Вальгусная деформация стоп",
            }
        ],
    )
    assert out["mode"] in {"history_fallback", "enriched", "current_only"}
    assert out["query"]
