"""Фильтр worst_severity и агрегат worst_severity_cases."""
from __future__ import annotations

from clinical_knowledge.mis_kz_quality import _filtered_agg, _match_filters


def test_worst_severity_exact_match() -> None:
    rec = {"p0": 0, "p1": 1, "p2": 2, "p3": 0}
    assert _match_filters(rec, {"worst_severity": "P1"})
    assert not _match_filters(rec, {"worst_severity": "P0"})
    assert not _match_filters(rec, {"worst_severity": "P2"})


def test_worst_severity_cases_agg() -> None:
    records = [
        {"p0": 1, "p1": 0, "p2": 0, "p3": 0},
        {"p0": 0, "p1": 2, "p2": 1, "p3": 0},
        {"p0": 0, "p1": 0, "p2": 1, "p3": 3},
        {"p0": 0, "p1": 0, "p2": 0, "p3": 1},
        {"p0": 0, "p1": 0, "p2": 0, "p3": 0},
    ]
    agg = _filtered_agg(records)
    assert agg["worst_severity_cases"] == {"P0": 1, "P1": 1, "P2": 1, "P3": 1, "none": 1}
