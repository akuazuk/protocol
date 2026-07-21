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


def test_brief_excludes_legal_boilerplate_from_chunks(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    rich = [
        {
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "page_from": 1,
            "text": (
                "Признать утратившим силу приказ Министерства здравоохранения Республики Беларусь. "
                "Национальный правовой Интернет-портал Республики Беларусь, 16.04.2022, 8/37875. "
                "Министр Д.Л.Пиневич СОГЛАСОВАНО Брестский областной исполнительный комитет."
            ),
        },
        {
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "page_from": 8,
            "text": "Базисная терапия проводится ингаляционными глюкокортикостероидами длительным курсом.",
        },
    ]
    brief = build_protocol_brief("x.pdf", rich_chunks=rich, title_hint="Протокол")
    treat = next(s for s in brief["sections"] if s["id"] == "treatment")
    texts = " ".join(p["text"] for p in treat["points"])
    assert "утратившим силу" not in texts
    assert "Интернет-портал" not in texts
    assert "Пиневич" not in texts
    assert "исполнительный комитет" not in texts
    assert "Базисная терапия" in texts


def test_brief_unavailable_without_data(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    brief = build_protocol_brief("x.pdf")
    assert brief["available"] is False
    assert brief["sections"] == []


def _expanded_summary() -> ProtocolSummary:
    cond = ConditionSummary(
        condition_id="j45",
        name="Бронхиальная астма",
        icd10_codes=["J45"],
        required_exams=[
            ExamRequirement(
                name="Спирометрия с бронходилатационным тестом",
                requirement_level="required",
                timing="при первичной диагностике",
                source_ref=_sr("Спирометрия с бронходилатационным тестом", 6),
            )
        ],
        treatment=TreatmentBlock(
            drugs=[
                DrugTreatmentItem(
                    drug_name="Будесонид",
                    route="ингаляционно",
                    dose_text="200-400 мкг",
                    frequency_text="2 раза в сутки",
                    duration_text="длительно",
                    indication="базисная терапия",
                    contraindications=["гиперчувствительность"],
                    source_ref=_sr("Будесонид 200-400 мкг ингаляционно 2 раза в сутки", 16),
                )
            ]
        ),
        red_flags=[
            RedFlagItem(
                text="Жизнеугрожающее обострение: SpO2 ниже 92% и немое лёгкое требуют реанимации.",
                severity="critical",
                expected_actions=["вызов реанимации", "кислород"],
                source_ref=_sr("Жизнеугрожающее обострение: SpO2 ниже 92% и немое лёгкое требуют реанимации.", 13),
            )
        ],
    )
    return ProtocolSummary(
        protocol_id="pulm_asthma2",
        source=ProtocolSource(
            title="Диагностика и лечение бронхиальной астмы у взрослых",
            local_path="p2.pdf",
        ),
        conditions=[cond],
    )


def test_brief_expands_drug_detail_and_tags(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _expanded_summary())
    brief = build_protocol_brief("p2.pdf", icd_codes=["J45"])
    treat = next(s for s in brief["sections"] if s["id"] == "treatment")
    pt = treat["points"][0]
    assert "Будесонид" in pt["text"]
    assert "препарат" in pt["tags"]
    labels = {d["label"]: d["value"] for d in pt["detail"]}
    assert labels.get("Доза") == "200-400 мкг"
    assert labels.get("Режим") == "2 раза в сутки"
    assert labels.get("Путь") == "ингаляционно"
    assert "Противопоказания" in labels


def test_brief_exam_requirement_and_redflag_severity(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _expanded_summary())
    brief = build_protocol_brief("p2.pdf", icd_codes=["J45"])
    exams = next(s for s in brief["sections"] if s["id"] == "exams")
    assert "обязательно" in exams["points"][0]["tags"]
    assert any(d["label"] == "Когда" for d in exams["points"][0]["detail"])
    rf = next(s for s in brief["sections"] if s["id"] == "red_flags")
    assert any(t.startswith("тяжесть:") for t in rf["points"][0]["tags"])
    assert any(d["label"] == "Действия" for d in rf["points"][0]["detail"])


def test_brief_entity_chips_from_card(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _expanded_summary())
    brief = build_protocol_brief("p2.pdf", icd_codes=["J45"])
    assert "Будесонид" in brief["entities"]["drugs"]


def test_brief_grounding_sets_verified(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _expanded_summary())

    def _lookup(_path, quote):
        return 16 if "Будесонид" in quote else None

    brief = build_protocol_brief("p2.pdf", icd_codes=["J45"], page_lookup=_lookup)
    treat = next(s for s in brief["sections"] if s["id"] == "treatment")
    assert treat["points"][0]["verified"] is True
    # красный флаг не найден lookup-ом -> не подтверждён (grounding строгий)
    rf = next(s for s in brief["sections"] if s["id"] == "red_flags")
    assert rf["points"][0]["verified"] is False


def test_brief_care_setting_backfill(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: _expanded_summary())
    brief = build_protocol_brief("p2.pdf", icd_codes=["J45"])
    # в тексте красного флага есть «реанимации» -> стационарный маршрут
    assert "Стационар" in brief["care_setting_labels"]


def test_brief_drops_title_echo_point(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    rich = [
        {
            "chunk_type": "diagnostics",
            "section_title": "Обследования",
            "page_from": 2,
            "text": (
                "КЛИНИЧЕСКИЙ ПРОТОКОЛ «Диагностика и лечение пациентов (взрослое население) "
                "с бронхиальной астмой». "
                "Общий анализ крови выполняется при первичном обследовании пациента."
            ),
        }
    ]
    brief = build_protocol_brief(
        "p.pdf",
        rich_chunks=rich,
        title_hint="Диагностика и лечение взрослое население с бронхиальной астмой",
    )
    exams = next(s for s in brief["sections"] if s["id"] == "exams")
    texts = " ".join(p["text"] for p in exams["points"])
    assert "КЛИНИЧЕСКИЙ ПРОТОКОЛ" not in texts
    assert "Общий анализ крови" in texts
