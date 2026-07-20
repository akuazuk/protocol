"""Трек 3: определение условий оказания (стационар/амбулаторно)."""
from __future__ import annotations

from clinical_knowledge.care_setting import (
    care_setting_label_ru,
    infer_care_setting_for_path,
    infer_care_setting_from_filename,
)


def test_inpatient_from_filename() -> None:
    p = "minzdrav_protocols/hirurgiya/КП_лечение_в_стационарных_условиях_2022.pdf"
    assert infer_care_setting_from_filename(p) == "inpatient"


def test_outpatient_from_filename() -> None:
    p = "minzdrav_protocols/terapiya/КП_ведение_в_амбулаторных_условиях.pdf"
    assert infer_care_setting_from_filename(p) == "outpatient"


def test_mixed_from_filename() -> None:
    p = "КП_помощь_в_амбулаторных_и_стационарных_условиях.pdf"
    assert infer_care_setting_from_filename(p) == "mixed"


def test_none_when_no_marker() -> None:
    assert infer_care_setting_from_filename("КП_диагностика_и_лечение.pdf") is None


def test_for_path_full_shape_filename() -> None:
    p = "КП_в_стационарных_условиях.pdf"
    out = infer_care_setting_for_path(p)
    assert out["care_setting"] == "inpatient"
    assert out["care_setting_label"] == "стационарно"
    assert out["care_setting_source"] == "filename"
    assert out["care_setting_confidence"] >= 0.7


def test_for_path_from_chunk_tags() -> None:
    chunks = [
        {"tags": {"care_setting": ["ambulatory"]}},
        {"tags": {"care_setting": ["ambulatory"]}},
        {"care_setting": ["амбулаторно"]},
    ]
    out = infer_care_setting_for_path(
        "КП_диагностика.pdf", "", chunks_getter=lambda _p: chunks
    )
    assert out["care_setting"] == "outpatient"
    assert out["care_setting_source"] == "chunk_tags"


def test_label_helper() -> None:
    assert care_setting_label_ru("inpatient") == "стационарно"
    assert care_setting_label_ru(None) is None
