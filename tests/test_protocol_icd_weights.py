"""Тесты рейтинга соответствия МКБ в protocol_catalog."""
from __future__ import annotations

from clinical_knowledge.protocol_catalog import (
    compute_icd10_relevance_weights,
    normalize_protocol_title,
)


def test_normalize_protocol_title_strips_date_and_kp() -> None:
    t = normalize_protocol_title(
        "КП Диагностика геморроя 12.03.2020 № 45",
        "КП_Диагностика_геморроя_12_03_2020.pdf",
    )
    assert "2020" not in t
    assert "№" not in t
    assert "геморро" in t.lower()


def test_icd_weights_primary_higher_than_secondary() -> None:
    w = compute_icd10_relevance_weights(
        icd_primary=["K64.9", "K62.5"],
        icd_all=["K64.9", "K62.5", "K92.2"],
        title="Геморрой и кровотечение",
        body="Лечение K64.9 и K62.5 при кровотечении",
    )
    assert w["K64.9"] >= w.get("K92.2", 0)
    assert w["K64.9"] >= 85
