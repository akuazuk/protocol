"""Unit tests for protocol summary navigation builder."""
from __future__ import annotations

from clinical_knowledge.protocol_summary.nav import build_protocol_summary_nav


def test_build_protocol_summary_nav_missing() -> None:
    out = build_protocol_summary_nav("minzdrav_protocols/unknown/no_such.pdf")
    assert out["available"] is False
    assert out["path"]
