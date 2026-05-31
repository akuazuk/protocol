"""Tests for protocol card registry builder."""
from __future__ import annotations

from corpus_pipeline.protocol_cards import (
    build_protocol_id,
    card_from_protocols_row,
    infer_approval_from_filename,
    infer_population_from_text,
)


def test_infer_population_adult_from_filename():
    fn = "КП_Диагностика_лечение_пациентов_вз_нас_заболеваниями_пищевода_пост_МЗ_2025_185.pdf"
    assert infer_population_from_text(fn) == "adult"


def test_infer_population_child_from_filename():
    fn = "КП_Диагностика_и_лечение_пациентов_дет_нас_с_гастроэзофагеальной_рефлюксной_болезнью.pdf"
    assert infer_population_from_text(fn) == "child"


def test_infer_approval_from_post_mz_filename():
    meta = infer_approval_from_filename(
        "КП_Диагностика_лечение_пациентов_вз_нас_заболеваниями_пищевода_желудка_двенадцатиперстной_кишки_пост_МЗ_2025_185.pdf"
    )
    assert meta.get("approval_number") == "185"
    assert meta.get("approval_date", "").startswith("2025")


def test_card_from_protocols_row_minimal():
    row = {
        "path": "minzdrav_protocols/gastroenterologiya/test.pdf",
        "category": "gastroenterologiya",
        "filename": "КП взрослые K21 пост_МЗ_2025_185.pdf",
        "title": "КП взрослые K21",
    }
    card = card_from_protocols_row(row)
    assert card["specialty_slug"] == "gastroenterologiya"
    assert card["population"] == "adult"
    assert "K21" in card["icd10_all"]


def test_build_protocol_id_stable():
    pid = build_protocol_id(
        "gastroenterologiya",
        "minzdrav_protocols/gastroenterologiya/foo.pdf",
        "L0",
        "185",
        "adult",
    )
    assert "gastroenterologiya" in pid
    assert "185" in pid
    assert pid.endswith("_l0")
