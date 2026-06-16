"""Дедупликация нозологий в protocol-summary-nav."""
from __future__ import annotations

from clinical_knowledge.protocol_summary.nav import dedupe_nav_conditions


def _cond(cid: str, name: str, icd: list[str], *, icd_match: bool = False) -> dict:
    return {
        "condition_id": cid,
        "name": name,
        "icd10_codes": icd,
        "icd_match": icd_match,
        "name_match": False,
        "sections": [{"id": "criteria", "label": "Критерии", "count": 2, "preview": "тест"}],
    }


def test_dedupe_same_icd_family_collapses_duplicates():
    rows = [
        _cond(
            "j20_0_x",
            "клинический протокол диагностики и лечения пиоторакса",
            ["J20"],
        ),
        _cond(
            "j20_1_x",
            "клинический протокол диагностики и лечения пиоторакса",
            ["J20"],
        ),
        _cond(
            "j41_x",
            "клинический протокол диагностики и лечения хронического бронхита предназначен",
            ["J41"],
        ),
    ]
    out = dedupe_nav_conditions(rows, query="J20.9 острый бронхит", icd_codes=["J20.9"])
    assert len(out) == 2
    assert out[0]["icd_match"] is True
    assert "Острый бронхит" in out[0]["display_label"]
    assert "J20" in out[0]["display_label"]
    assert out[0]["match_reason"]


def test_dedupe_relevance_score_orders_icd_match_first():
    rows = [
        _cond("j41_x", "хронический бронхит", ["J41"]),
        _cond("j20_x", "острый бронхит", ["J20"], icd_match=True),
    ]
    out = dedupe_nav_conditions(rows, query="J20.9", icd_codes=["J20.9"])
    assert out[0]["icd_match"] is True
    assert out[0]["relevance_score"] >= out[1]["relevance_score"]
