"""Быстрый индекс МКБ -> Protocol Summary."""
from __future__ import annotations

import time

from clinical_knowledge.protocol_summary.icd_index import (
    find_summary_refs_by_icd,
    prewarm_icd_summary_index,
)


def test_prewarm_icd_index_fast():
    t0 = time.perf_counter()
    n = prewarm_icd_summary_index()
    elapsed = time.perf_counter() - t0
    assert n >= 0
    assert elapsed < 2.0, f"ICD index build too slow: {elapsed:.2f}s"


def test_find_summary_refs_by_icd_returns_pairs():
    prewarm_icd_summary_index()
    refs = find_summary_refs_by_icd("K29.7", limit=3)
    assert isinstance(refs, list)
    if refs:
        assert len(refs[0]) == 2
