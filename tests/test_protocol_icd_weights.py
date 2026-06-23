"""Тесты рейтинга соответствия МКБ в protocol_catalog."""
from __future__ import annotations

from clinical_knowledge.protocol_catalog import (
    _icd_from_classification_sections,
    _is_external_cause_icd,
    compute_icd10_relevance_weights,
    normalize_protocol_title,
)


def test_is_external_cause_icd() -> None:
    assert _is_external_cause_icd("Y55.5")
    assert not _is_external_cause_icd("J02.9")


def test_icd_from_classification_sections_skips_pharmacotherapy() -> None:
    body = (
        "КЛАССИФИКАЦИЯ\n"
        "Острый фарингит J02.9, хронический ринит J30.4.\n"
        "ФАРМАКОТЕРАПИЯ\n"
        "Побочные эффекты Y55.5 при насморке.\n"
    )

    def _extract(text: str) -> list[str]:
        import re

        return re.findall(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", text or "")

    codes, src = _icd_from_classification_sections(
        body,
        extract_icd10=_extract,
        lookup_disease_icd=lambda _: [],
        prioritize_codes=lambda xs: list(xs),
        is_symptom_code=lambda _: False,
    )
    assert "J02.9" in codes
    assert "Y55.5" not in codes
    assert "body_classification_extract" in src


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
