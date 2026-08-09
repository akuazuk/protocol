"""R3: pediatric asthma KP ranks above rehab noise."""
from __future__ import annotations

from clinical_knowledge.case_protocol_suggest import _rank_rows, _rehab_or_noise_penalty


def test_rehab_penalty_higher_for_asthma() -> None:
    graph = {
        "icd10_in_directory": ["J45.0"],
        "diagnoses": [{"text": "Бронхиальная астма"}],
        "audience": "child",
    }
    rehab = {
        "source_path": "minzdrav_protocols/reabilitaciya/KP_asthma_rehab.pdf",
        "title": "Реабилитация при бронхиальной астме",
        "protocol_id": "rehab-asthma",
    }
    clinical = {
        "source_path": "minzdrav_protocols/allergologiya/KP_BA_d-nas_2025_38.pdf",
        "title": "Бронхиальная астма у детей пост. МЗ 2025 №38",
        "protocol_id": "ba-child-38",
    }
    assert _rehab_or_noise_penalty(rehab, graph) > _rehab_or_noise_penalty(clinical, graph)


def test_rank_rows_prefers_pediatric_asthma_over_rehab() -> None:
    graph = {
        "icd10_in_directory": ["J45"],
        "diagnoses": [{"text": "J45 бронхиальная астма"}],
        "audience": "child",
        "specialty": {"slug": "allergologiya"},
    }
    matched = [
        {
            "source_path": "minzdrav_protocols/reabilitaciya/KP_rehab_j45.pdf",
            "title": "Реабилитация астмы",
            "match_score": 90,
            "icd_fit": [{"code": "J45", "weight": 1.0}],
            "icd10_primary": ["J45"],
        },
        {
            "source_path": "minzdrav_protocols/allergologiya/KP_BA_d-nas_2025_38.pdf",
            "title": "Детская бронхиальная астма №38",
            "match_score": 80,
            "icd_fit": [{"code": "J45", "weight": 1.0}],
            "icd10_primary": ["J45"],
        },
        {
            "source_path": "minzdrav_protocols/pulmonologiya/KP_BA_adult.pdf",
            "title": "Бронхиальная астма у взрослых",
            "match_score": 85,
            "icd_fit": [{"code": "J45", "weight": 1.0}],
            "icd10_primary": ["J45"],
        },
    ]
    ranked = _rank_rows(matched, graph, limit=3, case_codes=["J45"])
    assert ranked
    top_path = (ranked[0].get("source_path") or "").lower()
    assert "d-nas" in top_path or "детск" in (ranked[0].get("title") or "").lower()
    assert "reabilit" not in top_path and "реабилитац" not in top_path
