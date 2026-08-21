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
    assert looks_omnibus(
        {
            "title": "Клинический протокол",
            "source_path": "кп_оториноларингология (взрослые) в ред. пост. мз рб от 25.06.2026 №78.pdf",
        }
    )
    assert looks_omnibus(
        {
            "title": "Диагностика и лечение взрослого населения с общехирургическими болезнями",
            "source_path": "khirurgiya/кп 12.02.2007 №82.pdf",
        }
    )
    assert looks_omnibus(
        {
            "title": "Диагностика и лечение пациентов взрослых с заболеваниями нейрохирургического профиля в стационарных условиях",
            "source_path": "nevrologiya/КП_2021_117.pdf",
        }
    )
    assert not looks_omnibus(
        {
            "title": "Медицинское наблюдение и оказание медицинской помощи женщинам в акушерстве и гинекологии",
            "source_path": "akusherstvo/КП_2018_№_17.pdf",
        }
    )
    assert not looks_omnibus(
        {"title": "Геморрой у взрослого населения", "source_path": "khirurgiya/kp22.pdf"}
    )
    assert not looks_omnibus(
        {
            "title": "Хронический синусит у взрослых",
            "source_path": "кп_диагностика_лечение_пациентов_в-нас_хроническим_синуситом_пост_мз_2025_25.pdf",
        }
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


def test_omnibus_blob_ignores_content_index(monkeypatch) -> None:
    from clinical_knowledge import protocol_content_index as pci
    from clinical_knowledge.protocol_match import _CARD_BLOB_CACHE, _card_match_blob

    _CARD_BLOB_CACHE.clear()
    monkeypatch.setattr(
        pci,
        "content_text_for_card",
        lambda _card: "острая инфекция верхних дыхательных путей орви насморк",
    )
    blob = _card_match_blob(
        {
            "title": "Диагностика и лечение пациентов с оториноларингологическими заболеваниями",
            "source_path": "minzdrav_protocols/otorinolaringologiya/КП_2017.pdf",
        }
    )
    assert "орви" not in blob
    assert "оториноларинголог" in blob
    sinus = _card_match_blob(
        {
            "title": "Хронический синусит у взрослых",
            "source_path": "кп_диагностика_лечение_пациентов_в-нас_хроническим_синуситом_пост_мз_2025_25.pdf",
        }
    )
    assert "орви" not in sinus
    assert "синусит" in sinus


def test_omnibus_ent_2026_revision_does_not_score_high() -> None:
    from clinical_knowledge.protocol_match import compute_match_score

    card = {
        "title": "Клинический протокол",
        "source_path": "кп_оториноларингология (взрослые) в ред. пост. мз рб от 25.06.2026 №78.pdf",
        "icd10_all": ["J06.9", "J32.9", "H66.9"],
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
