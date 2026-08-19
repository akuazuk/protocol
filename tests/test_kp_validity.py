from __future__ import annotations

from datetime import date

from clinical_knowledge.kp_validity import (
    attach_validity_fields,
    card_in_force_on,
    looks_omnibus,
    validity_from_card,
)


def test_in_force_excludes_cancelled_before_visit() -> None:
    card = {
        "title": "Урология 2011",
        "approval": {"date": "2011-08-09", "number": "920", "valid_to": "2015-01-01"},
        "status": "active",
    }
    assert card_in_force_on(card, date(2026, 7, 15)) is False
    assert card_in_force_on(card, date(2012, 6, 1)) is True


def test_sync_missing_without_date_is_not_hard_drop() -> None:
    card = {
        "title": "Старый файл",
        "status": "superseded",
        "superseded_by": "minzdrav_protocols/x/new.pdf",
        "approval": {"date": "2011-08-09"},
    }
    assert validity_from_card(card)["kind"] == "sync_missing"
    assert card_in_force_on(card, date(2026, 7, 15)) is True


def test_attach_validity_fills_valid_from_from_approval_date() -> None:
    card = attach_validity_fields({"approval": {"date": "2018-02-19", "number": "17"}})
    assert card["valid_from"] == "2018-02-19"
    assert card.get("valid_to") in (None, "")


def test_looks_omnibus_urology_and_dispanser() -> None:
    assert looks_omnibus(
        {
            "title": "Диагностика и лечение пациентов с урологическими заболеваниями",
            "source_path": "minzdrav_protocols/urologiya/КП 2011 №920.pdf",
        }
    )
    assert looks_omnibus({"title": "Диспансеризация взрослого населения", "source_path": "dn.pdf"})
    assert looks_omnibus(
        {
            "title": "Диагностика и лечение пациентов с оториноларингологическими заболеваниями",
            "source_path": "minzdrav_protocols/otorinolaringologiya/КП_2017.pdf",
        }
    )
    assert not looks_omnibus(
        {"title": "Геморрой у взрослого населения", "source_path": "khirurgiya/kp22.pdf"}
    )


def test_omnibus_ent_icd_dump_does_not_score_high() -> None:
    from clinical_knowledge.protocol_match import compute_match_score

    card = {
        "title": "Диагностика и лечение пациентов с оториноларингологическими заболеваниями",
        "source_path": "minzdrav_protocols/otorinolaringologiya/КП_2017.pdf",
        "icd10_all": ["J06.9", "J03.9", "H66.9", "J32.9", "J00"],
        "icd10_primary": ["J06.9"],
    }
    score = compute_match_score(
        card,
        icd_list=["J06.9"],
        audience="adult",
        hints=set(),
        specialty_slug=None,
        diag_text="Острая инфекция верхних дыхательных путей",
        complaints=[],
        performed_exams=[],
        use_icd=True,
    )
    assert score < 40, score
