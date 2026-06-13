"""Tests for corpus rule extraction heuristics."""
from __future__ import annotations

from clinical_knowledge.rules_from_corpus import (
    _parse_numbered_diagnosis_components,
    extract_rules_from_chunks,
    infer_condition_ids_from_source_path,
)


def test_infer_condition_from_path():
    sp = "minzdrav_protocols/gastroenterologiya/КП_язвенным_колитом.pdf"
    assert "ulcerative_colitis" in infer_condition_ids_from_source_path(sp)
    sp2 = "minzdrav_protocols/gastroenterologiya/КП_болезнью_Крона.pdf"
    assert "crohn" in infer_condition_ids_from_source_path(sp2)


def test_numbered_ibd_components():
    blob = """
    13.1. протяженность поражения кишечника: проктит;
    13.2. фазу течения - обострение или ремиссия: обострение;
    13.3. характер течения: хроническое;
    """
    comps = _parse_numbered_diagnosis_components(
        blob,
        ("13.1", "13.2", "13.3"),
    )
    assert "нозология" in comps
    assert "протяженность" in comps
    assert "фаза" in comps


def test_gastritis_hg_formula_with_line_breaks():
    text = (
        "Формулировка\nдиагноза\nХГ\nвключает:\n"
        "нозологическую форму;\nстепень активности;\nатрофия и метаплазия;"
    )
    chunk = {"text": text, "source_path": "minzdrav_protocols/gastroenterologiya/gastritis.pdf"}
    rules = extract_rules_from_chunks([chunk], protocol_id="t", rule_id_prefix="x")
    assert "gastritis" in rules
    assert any(r["rule_type"] == "diagnosis_formula" for r in rules["gastritis"])


def test_ibd_numbered_sections_from_path():
    blob = " ".join(
        [
            "13.1. протяженность поражения кишечника: проктит;",
            "13.2. фазу течения - обострение или ремиссия: ремиссия;",
            "13.4. тяжесть текущего обострения в соответствии с ПИАЯК: лёгкая;",
        ]
    )
    chunk = {
        "text": blob,
        "source_path": "minzdrav_protocols/gastroenterologiya/КП_язвенным_колитом.pdf",
    }
    rules = extract_rules_from_chunks(
        [chunk],
        protocol_id="t",
        rule_id_prefix="uc",
        source_path=chunk["source_path"],
    )
    assert "ulcerative_colitis" in rules
    formula = next(r for r in rules["ulcerative_colitis"] if r["rule_type"] == "diagnosis_formula")
    assert formula.get("extraction_method") == "numbered_sections"
    assert len(formula["required_components"]) >= 3
