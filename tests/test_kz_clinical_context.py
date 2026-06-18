"""Тесты клинического контекста амбулаторного КЗ."""
from __future__ import annotations

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.kz_clinical_context import (
    build_clinical_context,
    filter_ambulatory_kp_items,
    rank_kp_items_by_context,
    split_anamnesis_parts,
)

SAMPLE = """\
Жалобы: отёк левой ноги.
Анамнез заболевания: 5 дней назад появился отёк.
Анамнез жизни: варикоз, курение.
Диагноз: I80.1
"""


def test_split_anamnesis_parts():
    doc = parse_consultation(SAMPLE, consultation_id="ctx1")
    parts = split_anamnesis_parts(doc)
    assert "отёк" in parts["disease"].lower()
    assert "варикоз" in parts["life"].lower()


def test_build_clinical_context_ambulatory():
    doc = parse_consultation(SAMPLE, consultation_id="ctx2")
    ctx = build_clinical_context(doc, ["I80.1"])
    assert ctx["setting"] == "ambulatory"
    assert "I80.1" in ctx["icd_codes"]
    assert ctx["complaints"]
    assert "амбулатор" in ctx["clinical_query"].lower()


def test_filter_ambulatory_kp_items():
    items = [
        "УЗИ вен в амбулаторных условиях",
        "Госпитализация в стационар для операции",
        "КТ грудной клетки",
    ]
    out = filter_ambulatory_kp_items(items)
    assert "УЗИ" in out[0]
    assert not any("стационар" in x.lower() for x in out)


def test_rank_kp_by_complaints_and_icd():
    doc = parse_consultation(SAMPLE, consultation_id="ctx3")
    ctx = build_clinical_context(doc, ["I80.1"])
    ranked = rank_kp_items_by_context(
        [
            "МРТ головного мозга",
            "УЗИ вен нижних конечностей при тромбозе I80",
            "Госпитализация в круглосуточный стационар",
        ],
        ctx,
        limit=3,
    )
    texts = [r["text"] for r in ranked]
    assert texts[0].startswith("УЗИ")
    assert not any("круглосуточн" in t for t in texts)
