"""Фаза 1: серверный проектор карточек-выдержек протоколов (protocol_card)."""
from __future__ import annotations

import clinical_knowledge.protocol_summary.nav as nav_mod
from clinical_knowledge.protocol_card import (
    attach_protocol_cards,
    build_protocol_card,
)
from clinical_knowledge.protocol_summary.nav import build_protocol_card_from_summary
from clinical_knowledge.protocol_summary.schema import (
    ConditionSummary,
    CriteriaBlock,
    CriterionItem,
    DrugTreatmentItem,
    ExamRequirement,
    ProtocolSource,
    ProtocolSummary,
    RedFlagItem,
    SummarySourceRef,
    TreatmentBlock,
)


def _sr(quote: str, page: int | None = None) -> SummarySourceRef:
    return SummarySourceRef(quote=quote, page_start=page, section_title="Раздел")


def _sample_summary() -> ProtocolSummary:
    cond = ConditionSummary(
        condition_id="j18",
        name="Внебольничная пневмония",
        icd10_codes=["J18.9"],
        diagnostic_criteria=CriteriaBlock(
            required=[
                CriterionItem(
                    text="Лихорадка, кашель и инфильтрат на рентгенограмме грудной клетки.",
                    source_ref=_sr("Лихорадка, кашель и инфильтрат на рентгенограмме грудной клетки.", 12),
                )
            ]
        ),
        required_exams=[
            ExamRequirement(
                name="Рентгенография органов грудной клетки",
                requirement_level="required",
                source_ref=_sr("Рентгенография органов грудной клетки в двух проекциях.", 13),
            )
        ],
        treatment=TreatmentBlock(
            drugs=[
                DrugTreatmentItem(
                    drug_name="Амоксициллин",
                    source_ref=_sr("Амоксициллин 500 мг внутрь.", 20),
                )
            ]
        ),
        red_flags=[
            RedFlagItem(
                text="Сатурация ниже 92% - показание к госпитализации.",
                source_ref=_sr("Сатурация ниже 92% - показание к госпитализации.", 21),
            )
        ],
    )
    return ProtocolSummary(
        protocol_id="pulm_pneumonia",
        source=ProtocolSource(
            title="Внебольничная пневмония у взрослых",
            local_path="minzdrav_protocols/pulmonologiya/КП_пневмония.pdf",
        ),
        conditions=[cond],
    )


def test_card_from_summary_title_and_extracts(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _sample_summary())
    card = build_protocol_card_from_summary(
        "minzdrav_protocols/pulmonologiya/КП_пневмония.pdf",
        query="внебольничная пневмония",
        icd_codes=["J18.9"],
    )
    assert card["available"] is True
    assert card["source"] == "summary"
    assert card["title"] == "Внебольничная пневмония у взрослых"
    labels = [e["label"] for e in card["extracts"]]
    assert "Критерии и диагностика" in labels
    assert "Обследования" in labels
    assert "Лечение" in labels
    assert len(card["extracts"]) >= 2
    for e in card["extracts"]:
        assert e["text"]
        assert e["quote"]
    assert card["condition"]["icd_match"] is True


def test_card_extracts_carry_pages(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _sample_summary())
    card = build_protocol_card_from_summary("x", icd_codes=["J18.9"])
    pages = [e.get("page_start") for e in card["extracts"]]
    assert any(p for p in pages)


def _summary_without_pages() -> ProtocolSummary:
    cond = ConditionSummary(
        condition_id="j18",
        name="Внебольничная пневмония",
        icd10_codes=["J18.9"],
        diagnostic_criteria=CriteriaBlock(
            required=[
                CriterionItem(
                    text="Лихорадка и кашель с гнойной мокротой.",
                    source_ref=SummarySourceRef(quote="Лихорадка и кашель с гнойной мокротой."),
                )
            ]
        ),
    )
    return ProtocolSummary(
        protocol_id="pneumonia_nopage",
        source=ProtocolSource(title="Пневмония", local_path="p.pdf"),
        conditions=[cond],
    )


def test_page_lookup_enriches_missing_page(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _summary_without_pages())

    def _lookup(path, quote):
        assert "Лихорадка" in quote
        return 9

    card = build_protocol_card_from_summary("p.pdf", icd_codes=["J18.9"], page_lookup=_lookup)
    e = card["extracts"][0]
    assert e["page_start"] == 9
    assert e["page_source"] == "matched"


def test_page_source_summary_when_present(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _sample_summary())
    called = {"n": 0}

    def _lookup(path, quote):
        called["n"] += 1
        return 99

    card = build_protocol_card_from_summary("x", icd_codes=["J18.9"], page_lookup=_lookup)
    # У образца все source_ref уже содержат страницу -> lookup не вызывается, source=summary.
    assert called["n"] == 0
    for e in card["extracts"]:
        assert e["page_source"] == "summary"


def test_card_unavailable_when_no_summary(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    card = build_protocol_card_from_summary("missing.pdf")
    assert card["available"] is False
    assert card["source"] == "summary"


def test_build_card_fallback_to_rag(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    structured = {
        "sections": [
            {"kind": "criteria", "label": "Диагностика", "text": "Критерии диагноза.", "page_start": 3},
            {"kind": "treatment", "label": "Лечение", "text": "Схема лечения.", "page_start": 8},
        ]
    }
    card = build_protocol_card(
        "p.pdf",
        structured_excerpt=structured,
        title_hint="Протокол X",
    )
    assert card["available"] is True
    assert card["source"] == "rag"
    assert card["title"] == "Протокол X"
    assert len(card["extracts"]) == 2
    assert card["extracts"][0]["page_start"] == 3


def test_build_card_fallback_to_raw(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    card = build_protocol_card(
        "p.pdf",
        raw_excerpt="Некоторый фрагмент текста протокола.",
        title_hint="Протокол Y",
    )
    assert card["available"] is True
    assert card["source"] == "raw"
    assert card["extracts"][0]["label"] == "Фрагмент"


def test_build_card_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    card = build_protocol_card("p.pdf", title_hint="Пусто")
    assert card["available"] is False
    assert card["source"] is None


def test_attach_protocol_cards_populates_payload(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _sample_summary())
    payload = {
        "llm_json": {
            "protocols": [
                {"path": "minzdrav_protocols/pulmonologiya/КП_пневмония.pdf", "title": "Пневмония"},
            ]
        },
        "protocol_ui_meta": {
            "minzdrav_protocols/pulmonologiya/КП_пневмония.pdf": {"care_setting_label": "стационарно"}
        },
        "icd": {"codes_for_retrieval": ["J18.9"]},
    }
    out = attach_protocol_cards(payload, [], query="пневмония")
    assert "protocol_card" in out
    card = out["protocol_card"]["minzdrav_protocols/pulmonologiya/КП_пневмония.pdf"]
    assert card["available"] is True
    assert card["source"] == "summary"


def test_attach_uses_raw_when_no_summary(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    payload = {
        "llm_json": {"protocols": [{"path": "a.pdf", "title": "A"}]},
    }
    retrieval = [
        {"path": "a.pdf", "excerpt": "Короткий."},
        {"path": "a.pdf", "excerpt": "Более длинный осмысленный фрагмент протокола."},
    ]
    out = attach_protocol_cards(payload, retrieval, query="q")
    card = out["protocol_card"]["a.pdf"]
    assert card["source"] == "raw"
    assert "длинный" in card["extracts"][0]["text"]
