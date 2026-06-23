"""Шаг 2 воронки: коды МКБ для UI."""
from __future__ import annotations

from clinical_knowledge.search_funnel import _icd_payload_from_codes, handle_search_funnel


def test_icd_payload_from_codes_builds_choices() -> None:
    payload, choices = _icd_payload_from_codes(["J02.9"])
    assert payload["codes_for_retrieval"] == ["J02.9"]
    assert choices
    assert choices[0]["id"] == "J02.9"


def test_funnel_step2_confirmed_icd_returns_choices() -> None:
    out = handle_search_funnel(
        query="боль в горле и температура",
        step=2,
        context={"icd_codes": ["J02.9"], "icd_confirmed": True},
        category_slugs=[],
        session_id="test",
    )
    assert out.get("auto_skip") is True
    assert out.get("next_step") == 3
    assert len(out.get("choices") or []) >= 1
    icd = out.get("icd") or {}
    assert "J02.9" in (icd.get("codes_for_retrieval") or [])
