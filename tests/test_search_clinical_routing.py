"""Tests for clinical routing in protocol search."""
from clinical_knowledge.search_clinical_routing import (
    detect_clinical_route_ids,
    score_path_for_clinical_routes,
)


def test_detect_pregnancy_route():
    ids = detect_clinical_route_ids("беременность 32 недели головная боль", ["O26"])
    assert "pregnancy" in ids


def test_detect_hypertension_route():
    ids = detect_clinical_route_ids("гипертоническая болезнь давление 170/100", [])
    assert "hypertension" in ids


def test_score_burn_over_urology():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav/khirurgiya/ozhog.pdf",
        "КП диагностики лечения термических ожогов",
        route_ids=["burn"],
    )
    assert delta > 0
    delta2, _ = score_path_for_clinical_routes(
        "minzdrav/urologiya/cystitis.pdf",
        "урологические заболевания",
        route_ids=["burn"],
    )
    assert delta2 < 0


def test_score_lupus_not_dermatology_bullous():
    delta, _ = score_path_for_clinical_routes(
        "x/bullous.pdf",
        "буллезные нарушения",
        route_ids=["lupus"],
    )
    assert delta < 0
