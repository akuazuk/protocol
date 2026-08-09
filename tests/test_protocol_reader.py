"""Study-reader: абзацы ≥280 / 2 предложения, не brief-пункты."""
from __future__ import annotations

import clinical_knowledge.protocol_summary.nav as nav_mod
from clinical_knowledge.protocol_reader import build_protocol_reader
from clinical_knowledge.protocol_summary.schema import (
    ConditionSummary,
    DrugTreatmentItem,
    ProtocolSource,
    ProtocolSummary,
    SummarySourceRef,
    TreatmentBlock,
)


def _long(text: str) -> str:
    """Добить абзац до hard-min без смены смысла."""
    base = text.strip()
    while len(base) < 280:
        base += " Контроль эффективности терапии оценивают по динамике симптомов и переносимости."
    return base


def test_reader_builds_paragraphs_from_rich_chunks(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)
    rich = [
        {
            "chunk_type": "criteria",
            "section_title": "Критерии",
            "page_from": 2,
            "text": _long(
                "Диагноз астмы устанавливают по клинической картине и обратимой бронхиальной обструкции. "
                "Подтверждение требует спирометрии с бронходилатационной пробой."
            ),
        },
        {
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "page_from": 8,
            "text": _long(
                "Базисная терапия проводится ингаляционными глюкокортикостероидами длительным курсом. "
                "При недостаточном контроле добавляют длительно действующие бета-2-агонисты."
            ),
            "drugs": ["Будесонид"],
        },
        {
            "chunk_type": "treatment",
            "section_title": "Лечение",
            "page_from": 1,
            "text": (
                "Признать утратившим силу приказ Министерства здравоохранения Республики Беларусь. "
                "Национальный правовой Интернет-портал Республики Беларусь, 16.04.2022, 8/37875."
            ),
        },
        {
            "chunk_type": "treatment",
            "section_title": "Короткий",
            "page_from": 9,
            "text": "Короткий обрывок без контекста.",
        },
    ]
    out = build_protocol_reader("x.pdf", rich_chunks=rich, query="астма")
    assert out["available"] is True
    assert out["source"] == "rich_chunks"
    assert out["stats"]["paragraphs"] >= 2
    assert out["stats"]["median_len"] >= 180
    texts = " ".join(p["text"] for p in out["paragraphs"])
    assert "утратившим силу" not in texts
    assert "Короткий обрывок" not in texts
    assert "Базисная терапия" in texts
    treat = [p for p in out["paragraphs"] if p["section_id"] == "treatment"]
    assert treat
    assert treat[0]["page_start"] == 8
    assert "Будесонид" in (treat[0]["entities"].get("drugs") or [])
    assert any(t["id"] == "treatment" for t in out["toc"])


def test_reader_annotations_from_summary(monkeypatch) -> None:
    summary = ProtocolSummary(
        protocol_id="asthma",
        source=ProtocolSource(
            title="Астма",
            local_path="minzdrav_protocols/pulmonologiya/КП_астма.pdf",
        ),
        conditions=[
            ConditionSummary(
                condition_id="j45",
                name="Астма",
                icd10_codes=["J45"],
                treatment=TreatmentBlock(
                    drugs=[
                        DrugTreatmentItem(
                            drug_name="Будесонид",
                            dose_text="200-400 мкг",
                            frequency_text="2 р/сут",
                            source_ref=SummarySourceRef(quote="Будесонид", page_start=8),
                        )
                    ]
                ),
            )
        ],
    )
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: summary)
    rich = [
        {
            "chunk_type": "treatment",
            "page_from": 8,
            "text": _long(
                "Ингаляционный будесонид применяют как базисную терапию бронхиальной астмы у взрослых. "
                "Дозу титруют по контролю симптомов и функции внешнего дыхания."
            ),
            "drugs": ["Будесонид"],
        }
    ]
    out = build_protocol_reader(
        "minzdrav_protocols/pulmonologiya/КП_астма.pdf",
        rich_chunks=rich,
    )
    anns = out["paragraphs"][0]["annotations"]
    assert anns
    assert any("Будесонид" in (a.get("label") or "") for a in anns)


def test_reader_unavailable_without_data(monkeypatch) -> None:
    monkeypatch.setattr(nav_mod, "find_summary_by_catalog_path", lambda _p: None)

    def _empty(_path, **_kw):
        return {"available": False, "path": _path, "sections": {}}

    monkeypatch.setattr(
        "clinical_knowledge.protocol_summary.source_text.resolve_protocol_source_text",
        _empty,
    )
    out = build_protocol_reader("missing.pdf", rich_chunks=None)
    assert out["available"] is False
    assert out["paragraphs"] == []
