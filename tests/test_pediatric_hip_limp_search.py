"""Тесты: хромота/боль в бедре у ребёнка - МКБ и clinical expander."""
from __future__ import annotations

from clinical_knowledge.search_clinical_routing import detect_clinical_route_ids
from clinical_knowledge.search_query_expand import (
    expand_clinical_query_terms,
    is_pediatric_hip_limp_complaint,
    needs_msk_clinical_clarify,
)
from icd_mkb import analyze_query_for_icd, clinical_hints_confident, suggest_icd_from_russian


HIP_LIMP_Q = "боль в бедре больше месяца у ребенка 9 лет и хромота"


def test_pediatric_hip_limp_detected() -> None:
    assert is_pediatric_hip_limp_complaint(HIP_LIMP_Q)
    assert needs_msk_clinical_clarify(HIP_LIMP_Q, {})


def test_pediatric_hip_limp_icd_not_femoral_noise() -> None:
    codes = [s["code"] for s in suggest_icd_from_russian(HIP_LIMP_Q, max_results=8)]
    assert codes, "ожидались коды МКБ"
    top = codes[:4]
    assert any(c.startswith(("M91", "M08", "M25", "R26", "M00", "M13")) for c in top)
    for bad in ("G57.2", "I80.1", "K41", "R95"):
        assert bad not in codes[:6], f"шум {bad} в top: {codes}"


def test_pediatric_hip_limp_analyze_codes_for_retrieval() -> None:
    analysis = analyze_query_for_icd(HIP_LIMP_Q, HIP_LIMP_Q)
    codes = analysis.get("codes_for_retrieval") or []
    assert codes
    assert any(c.startswith(("M91", "M08", "M25", "R26")) for c in codes[:4])
    assert "R95" not in codes
    assert "G57.2" not in codes
    assert clinical_hints_confident(HIP_LIMP_Q)


def test_expand_adds_perthes_and_hip_joint() -> None:
    expanded, meta = expand_clinical_query_terms(HIP_LIMP_Q)
    assert meta.get("applied") is True
    low = expanded.lower().replace("ё", "е")
    assert "тазобедрен" in low
    assert "пертес" in low or "коксит" in low


def test_pediatric_hip_route_active() -> None:
    routes = detect_clinical_route_ids(HIP_LIMP_Q)
    assert "pediatric_hip_limp" in routes
