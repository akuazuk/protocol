"""C2: audience filter inside retrieve() from funnel context."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def retrieve_fn():
    from rag_server import retrieve

    return retrieve


def test_audience_filter_excludes_pediatric_title_for_adult_context(retrieve_fn):
    out = retrieve_fn(
        "кашель\nКонтекст подбора: взрослое население",
        max_chunks=8,
        max_per_path=2,
        routing_query="кашель\nКонтекст подбора: взрослое население",
    )
    if not out:
        pytest.skip("mini corpus has no audience-marked chunks")
    for row in out:
        p = (row.get("path") or "") + " " + (row.get("title") or "")
        pl = p.lower()
        assert "детск" not in pl or "взросл" in pl
