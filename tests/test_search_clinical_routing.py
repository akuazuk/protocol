"""Tests for clinical routing in protocol search."""
from clinical_knowledge.search_clinical_routing import (
    detect_clinical_route_ids,
    score_path_for_clinical_routes,
)


def test_detect_pregnancy_route():
    ids = detect_clinical_route_ids(
        "беременность 32 недели головная боль отёки\nКонтекст подбора: беременные",
        [],
    )
    assert "pregnancy" in ids
    assert "migraine" not in ids
    assert "heart_failure" not in ids


def test_score_pregnancy_penalizes_pediatric_neurology():
    bad_path = (
        "minzdrav_protocols/nevrologiya-neyrokhirurgiya/"
        "КП_Диагностика_лечение_пациентов_заболеваниями_нервной_системы_детс_нас"
    )
    delta, _ = score_path_for_clinical_routes(
        bad_path,
        "КП нервной системы детс нас",
        route_ids=["pregnancy"],
    )
    assert delta < -10


def test_detect_orvi_route():
    ids = detect_clinical_route_ids("J06.9 ОРВИ насморк кашель", ["J06.9"])
    assert "orvi_uri" in ids


def test_detect_hypertension_route():
    ids = detect_clinical_route_ids("гипертоническая болезнь давление 170/100", [])
    assert "hypertension" in ids


def test_detect_gastro_route_bloating_constipation():
    ids = detect_clinical_route_ids(
        "вздутие живота и запоры\nКонтекст подбора: взрослое население",
        ["K59.0"],
    )
    assert "gastroenterology" in ids


def test_score_gastro_penalizes_abdominal_trauma():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav_protocols/khirurgiya/kp_trauma_abdomen.pdf",
        "КП диагностика лечение пациентов с травмой живота",
        route_ids=["gastroenterology"],
    )
    assert delta < -10


def test_score_gastro_boosts_intestinal():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav/gastro/intestine.pdf",
        "КП заболеваниями кишечника",
        route_ids=["gastroenterology"],
    )
    assert delta > 0


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


def test_detect_coronary_route_icd():
    ids = detect_clinical_route_ids("", ["I25"])
    assert "coronary" in ids


def test_score_coronary_penalizes_ent_hub():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav_protocols/otorinolaringologiya/ent_hub.pdf",
        "КП оториноларингологическими заболеваниями в-нас",
        route_ids=["coronary"],
    )
    assert delta < -10


def test_score_hypertension_penalizes_pediatric_cardiology():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav/bolezni-sistemy-krovoobrashcheniya/ped_cardio.pdf",
        "КП кардиологическими заболеваниями д-нас",
        route_ids=["hypertension"],
    )
    assert delta < -10


def test_score_burn_penalizes_abdominal_trauma():
    delta, _ = score_path_for_clinical_routes(
        "minzdrav/khirurgiya/abd_trauma.pdf",
        "КП травма живота в стационарных условиях",
        route_ids=["burn"],
    )
    assert delta < -10
