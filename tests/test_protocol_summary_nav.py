"""Unit tests for protocol summary navigation builder."""
from __future__ import annotations

from clinical_knowledge.protocol_summary.nav import (
    build_protocol_summary_nav,
    build_section_excerpt,
)


def test_build_protocol_summary_nav_missing() -> None:
    out = build_protocol_summary_nav("minzdrav_protocols/unknown/no_such.pdf")
    assert out["available"] is False
    assert out["path"]


def test_build_section_excerpt_missing_protocol() -> None:
    out = build_section_excerpt(
        "minzdrav_protocols/unknown/no_such.pdf",
        condition_id="c1",
        section_id="criteria",
    )
    assert out["available"] is False
    assert out.get("llm_used") is False
