"""B3: CI eval по tests/fixtures/search_golden.jsonl."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clinical_knowledge.search_golden_eval import (
    DEFAULT_GOLDEN,
    evaluate_search_golden_row,
    load_search_golden,
    summarize_search_golden,
    write_snapshot,
)


def _is_mini_corpus() -> bool:
    p = os.environ.get("RAG_CHUNKS_JSONL", "")
    return "chunks.mini.jsonl" in p


@pytest.fixture(scope="module")
def retrieve_fn():
    from rag_server import retrieve

    return retrieve


def test_search_golden_fixture_schema():
    rows = load_search_golden()
    assert len(rows) >= 30
    types = {str(r.get("query_type")) for r in rows}
    assert "symptom" in types and "icd" in types and "mixed" in types
    for r in rows:
        assert r.get("query")
        assert "funnel_step" in r
        if not r.get("expect_empty"):
            assert r.get("expected_path_contains") or r.get("expected_path")


def test_search_golden_mini_corpus_hit3(retrieve_fn):
    rows = [r for r in load_search_golden() if r.get("corpus") == "mini"]
    assert len(rows) >= 5
    reports = [evaluate_search_golden_row(r, retrieve_fn) for r in rows]
    summary = summarize_search_golden(reports)
    assert summary["n"] == len(rows)
    assert summary["hit3"] is not None
    assert summary["hit3"] >= 0.6, summary


def test_search_golden_full_corpus_optional(retrieve_fn):
    if _is_mini_corpus() and os.environ.get("SEARCH_GOLDEN_FULL") != "1":
        pytest.skip("full corpus golden: set SEARCH_GOLDEN_FULL=1")
    rows = [r for r in load_search_golden() if r.get("corpus") == "full"]
    if not rows:
        pytest.skip("no full corpus rows")
    reports = []
    for r in rows[:12]:
        reports.append(evaluate_search_golden_row(r, retrieve_fn))
    summary = summarize_search_golden(reports)
    assert summary["n"] >= 10


def test_search_golden_snapshot_artifact(retrieve_fn, tmp_path: Path):
    rows = [r for r in load_search_golden() if r.get("corpus") == "mini"]
    reports = [evaluate_search_golden_row(r, retrieve_fn) for r in rows]
    summary = summarize_search_golden(reports)
    summary["source"] = str(DEFAULT_GOLDEN)
    summary["corpus"] = "mini"
    if os.environ.get("SEARCH_GOLDEN_WRITE_SNAPSHOT") == "1":
        write_snapshot(summary)
    else:
        out = tmp_path / "search_golden_snapshot.json"
        write_snapshot(summary, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["hit3"] is not None
