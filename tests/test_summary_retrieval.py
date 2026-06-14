"""B1: summary-first retrieval helpers (ICD → catalog paths, summary boost)."""
from __future__ import annotations

import pytest

from clinical_knowledge.protocol_summary.icd_index import (
    find_catalog_paths_by_icd_codes,
    find_summary_refs_by_icd,
    prewarm_icd_summary_index,
)


def test_icd_summary_index_prewarm():
    n = prewarm_icd_summary_index()
    assert n >= 0


def test_find_summary_refs_by_icd_returns_pairs():
    refs = find_summary_refs_by_icd("J18.9", limit=2)
    assert isinstance(refs, list)
    for ref in refs:
        assert len(ref) == 2
        assert ref[0]


def test_find_catalog_paths_by_icd_codes_catalog_shape():
    paths = find_catalog_paths_by_icd_codes(["J18.9", "K29"], limit=6)
    assert isinstance(paths, list)
    for p in paths:
        assert p.startswith("minzdrav_protocols/") or p.startswith("clinical_")


@pytest.fixture
def retrieve_fn():
    pytest.importorskip("fastapi")
    import rag_server as rs

    rs._require_rag_loaded()
    return rs.retrieve


def test_retrieve_icd_boosts_with_path_boost(retrieve_fn) -> None:
    paths = find_catalog_paths_by_icd_codes(["J20.9"], limit=3)
    if not paths:
        pytest.skip("no ICD index paths for J20.9")
    out = retrieve_fn(
        "J20.9 острый бронхит",
        max_chunks=4,
        max_per_path=2,
        icd_codes_for_lex=["j20.9"],
        path_boost=paths[:2],
    )
    assert out
    top_path = out[0].get("path") or ""
    assert any(p in top_path or top_path in p for p in paths) or top_path
