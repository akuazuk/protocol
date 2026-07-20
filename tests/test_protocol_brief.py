"""Единая сводка протокола (protocol_brief): выводы по разделам без дублей и обрывков."""
from __future__ import annotations

import clinical_knowledge.protocol_summary.nav as nav_mod
from clinical_knowledge.protocol_brief import build_protocol_brief
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


def _summary() -> ProtocolSummary:
    cond = ConditionSummary(
        condition_id="j45",
        name="Бронхиальная астма",
        icd10_codes=["J45"],
        diagnostic_criteria=CriteriaBlock(
            required=[
                CriterionItem(
                    text="Диагноз астмы устанавливают по клинической картине и обратимой бронхиальной обструкции.",
                    source_ref=_sr(
                        "Диагноз астмы устанавливают по клинической картине и обратимой бронхиальной обструкции.",
                        5,
                    ),
                ),
                CriterionItem(
                    text="Диагноз астмы устанавливают по клинической картине и обратимой обструкции бронхов.",
                    source_ref=_sr("почти дубль первой формулировки критерия", 5),
                ),
            ]
        ),
        required_exams=[
            ExamRequirement(
                name="Спирометрия с бронходилатационной пробой для оценки обратимости обструкции.",
                requirement_level="required",
                source_ref=_sr("Спирометрия с бронходилатационной пробой для оценки обратимости обструкции.", 6),
            )
        ],
        treatment=TreatmentBlock(
            drugs=[
                DrugTreatmentItem(
                    drug_name="Будесонид",
                    source_ref=_sr(
                        "Ингаляционные глюкокортикостероиды являются базисной терапией бронхиальной астмы.",
                        8,
                    ),
                )
            ]
        ),
        red_flags=[
            RedFlagItem(
                text="Тяжёлое обострение с гипоксией - показание к неотложной госпитализации пациента.",
                source_ref=_sr("Тяжёлое обострение с гипоксией - показание к неотложной госпитализации пациента.", 9),
            )
        ],
    )
    return ProtocolSummary(
        protocol_id="pulm_asthma",
        source=ProtocolSource(
            title="Диагностика и лечение бронхиальной астмы у взрослых",
            local_path="minzdrav_protocols/pulmonologiya/КП_астма.pdf",
        ),
        conditions=[cond],
    )


def test_brief_sections_and_dedup(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _summary())
    brief = build_protocol_brief(
        "minzdrav_protocols/pulmonologiya/КП_астма.pdf",
        query="бронхиальная астма",
        icd_codes=["J45"],
    )
    assert brief["available"] is True
    assert brief["source"] == "summary"
    assert brief["title"].startswith("Диагностика и лечение")
    ids = [s["id"] for s in brief["sections"]]
    assert "diagnosis" in ids and "treatment" in ids
    diag = next(s for s in brief["sections"] if s["id"] == "diagnosis")
    # near-дубль критерия схлопнут -> ровно одна точка
    assert diag["count"] == 1
    assert diag["points"][0]["verified"] is True
    assert diag["points"][0]["page_start"] == 5


def test_brief_falls_back_to_rich_chunks(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    rich = [
        {
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "page_from": 8,
            "text": (
                "Базисная терапия проводится ингаляционными глюкокортикостероидами. "
                "При недостаточном контроле добавляют длительно действующие бета-2-агонисты."
            ),
            "drugs": ["Будесонид", "Формотерол"],
        }
    ]
    brief = build_protocol_brief(
        "x.pdf",
        rich_chunks=rich,
        title_hint="Протокол по астме",
    )
    assert brief["available"] is True
    assert brief["source"] == "rich_chunks"
    treat = next(s for s in brief["sections"] if s["id"] == "treatment")
    assert treat["count"] == 2
    assert brief["entities"]["drugs"]
    assert brief["full_text_available"] is True


def test_brief_filters_glossary_and_icd_shifr(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    rich = [
        {
            "chunk_type": "criteria",
            "section_title": "Критерии",
            "page_from": 2,
            "text": (
                "АСИТ - аллергенспецифическая иммунотерапия. "
                "(шифр по Международной статистической классификации болезней - J45). "
                "Диагноз астмы подтверждают спирометрией с пробой на обратимость обструкции."
            ),
        }
    ]
    brief = build_protocol_brief("x.pdf", rich_chunks=rich, title_hint="Астма")
    diag = next(s for s in brief["sections"] if s["id"] == "diagnosis")
    texts = " ".join(p["text"] for p in diag["points"])
    assert "АСИТ" not in texts
    assert "шифр" not in texts
    assert "спирометрией" in texts


def test_brief_unavailable_without_data(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    brief = build_protocol_brief("x.pdf")
    assert brief["available"] is False
    assert brief["sections"] == []
