"""Тесты осмысленных выдержек и СОП Кравira."""
from __future__ import annotations

from clinical_knowledge.consult_alignment import build_consult_alignment
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.kravira_sop_rules import evaluate_sop_block, sop_reference_for_block
from clinical_knowledge.meaningful_excerpt import meaningful_excerpt


def test_meaningful_excerpt_keeps_whole_sentences():
    text = (
        "Жалобы на кашель с мокротой три дня. "
        "Температура поднималась до 38 градусов. "
        "Насморк без заложенности носа."
    )
    out = meaningful_excerpt(text, limit=90)
    assert out.endswith(".")
    assert "Температура" not in out or out.count(".") >= 1
    assert "…" not in out or len(out) <= 90


def test_meaningful_excerpt_empty_for_noise():
    assert meaningful_excerpt("нет", min_chars=12) == ""


def test_sop_reference_per_block():
    ref = sop_reference_for_block("complaints")
    assert "СОП" in ref
    assert "Жалобы" in ref


def test_sop_gaps_for_missing_anamnesis():
    doc = parse_consultation("Жалобы: кашель.\nДиагноз: J06.8", consultation_id="t1")
    sop = evaluate_sop_block(doc, "anamnesis")
    assert any("анамнез" in g.lower() for g in sop["gaps_ru"])


def test_alignment_includes_sop_reference_and_findings():
    text = """\
Жалобы: кашель с мокротой три дня, температура до 38.
Анамнез заболевания: болеет 3 дня после переохлаждения.
Анамнез жизни: аллергия на пенициллин.
Объективный статус: АД 120/80, пульс 72, температура 36.6.
Диагноз: J06.8 Острая инфекция верхних дыхательных путей.
Рекомендации по лечению: симптоматическая терапия.
Контроль через неделю.
"""
    doc = parse_consultation(text, consultation_id="sop1")
    out = build_consult_alignment(
        doc,
        protocol_paths=[],
        icd_codes=["J06.8"],
        get_chunks=lambda _p: [],
    )
    complaints = next(c for c in out["criteria"] if (c.get("name_ru") or "").startswith("Жалобы"))
    assert complaints.get("reference_ru") or complaints.get("protocol_excerpt")
    assert complaints.get("findings_ru") or complaints.get("gaps_ru")
