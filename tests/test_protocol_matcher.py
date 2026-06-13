"""Тесты применимости и подбора протоколов (ТЗ раздел 24)."""
from __future__ import annotations

from clinical_knowledge.applicability import assess_card_applicability
from clinical_knowledge.protocol_match import (
    annotate_applicability,
    match_protocol_cards,
)


def test_child_protocol_not_applied_to_adult():
    card = {"title": "КП детский гастрит", "population": "child", "source_path": "x/ped.pdf"}
    appl, _, mismatch = assess_card_applicability(card, {"adult_or_child": "adult", "age_years": 40})
    assert appl == "not_applicable"
    assert mismatch


def test_pediatric_title_marker_not_applied_to_adult():
    card = {
        "title": "Диагностика и лечение пациентов (детское население) Неврология",
        "population": "any",
        "source_path": "nevrologiya/ped_neuro.pdf",
    }
    appl, _, mismatch = assess_card_applicability(card, {"adult_or_child": "adult", "age_years": 48})
    assert appl == "not_applicable"
    assert mismatch


def test_match_protocol_cards_skip_pediatric_for_adult():
    facts = {
        "patient_context": {"adult_or_child": "adult", "age_years": 48},
        "consultation": {"icd10": ["M54.1"], "conditions_hint": [], "diagnosis_text": "радикулопатия"},
    }
    res = match_protocol_cards(facts, limit=12)
    if not res:
        import pytest

        pytest.skip("Реестр карточек протоколов не загружен в этом окружении.")
    for m in res:
        blob = ((m.get("title") or "") + " " + (m.get("source_path") or "")).lower()
        assert "детск" not in blob or "взросл" in blob


def test_pregnancy_protocol_not_applied_without_pregnancy():
    card = {"title": "КП ведение беременности", "population": "adult", "source_path": "akush/preg.pdf"}
    appl, _, mismatch = assess_card_applicability(
        card, {"adult_or_child": "adult", "sex": "female", "pregnancy": None}
    )
    assert appl in ("possibly_applicable", "not_applicable")
    assert mismatch
    # мужчина - точно неприменимо
    appl_m, _, _ = assess_card_applicability(
        card, {"adult_or_child": "adult", "sex": "male", "pregnancy": None}
    )
    assert appl_m == "not_applicable"


def test_pregnancy_protocol_applicable_when_pregnant():
    card = {"title": "КП беременность", "population": "adult", "source_path": "akush/preg.pdf"}
    appl, match_reasons, _ = assess_card_applicability(
        card, {"adult_or_child": "adult", "sex": "female", "pregnancy": True}
    )
    assert appl == "applicable"
    assert match_reasons


def test_any_population_applicable_to_adult():
    card = {"title": "КП гастрит", "population": "any", "source_path": "gastro/g.pdf"}
    appl, _, _ = assess_card_applicability(card, {"adult_or_child": "adult", "age_years": 50})
    assert appl == "applicable"


def test_unknown_patient_is_not_blocking():
    card = {"title": "КП", "population": "adult", "source_path": "x.pdf"}
    appl, _, _ = assess_card_applicability(card, {})
    assert appl in ("possibly_applicable", "unknown")


def test_annotate_applicability_additive():
    matches = [{"protocol_id": "p1", "population": "child", "title": "детский", "source_path": "a.pdf"}]
    out = annotate_applicability(matches, {"adult_or_child": "adult"})
    assert out[0]["protocol_id"] == "p1"  # исходные поля сохранены
    assert out[0]["applicability"] == "not_applicable"
    assert "applicability" not in matches[0]  # исходник не мутирован


def test_icd_matching_gastro_smoke():
    """K21/K30 должны вытаскивать гастро-протоколы (если реестр загружен)."""
    facts = {
        "patient_context": {"adult_or_child": "adult"},
        "consultation": {"icd10": ["K21.0"], "conditions_hint": ["gerd"]},
    }
    res = match_protocol_cards(facts, limit=10)
    if not res:
        import pytest

        pytest.skip("Реестр карточек протоколов не загружен в этом окружении.")
    blob = " ".join((m.get("source_path") or "") + (m.get("title") or "") for m in res).lower()
    assert any(x in blob for x in ("gastro", "гэрб", "рефлюкс", "пищевод", "желуд"))
