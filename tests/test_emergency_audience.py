"""Аудитория воронки: неотложная помощь без детского контекста → взрослые."""
from __future__ import annotations

from rag_server import infer_audience_from_funnel_context


def test_emergency_funnel_defaults_to_adult() -> None:
    q = "анафилаксия\nКонтекст подбора: неотложная помощь"
    assert infer_audience_from_funnel_context(q) == "adult"


def test_emergency_funnel_pediatric_when_child_in_query() -> None:
    q = "инородное тело у ребёнка\nКонтекст подбора: неотложная помощь"
    assert infer_audience_from_funnel_context(q) == "child"
