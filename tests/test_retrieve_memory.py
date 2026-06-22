"""Retrieve memory: path allowlist не сканирует весь корпус."""
from __future__ import annotations

import rag_server as rs


def test_chunk_indices_for_path_allowlist(monkeypatch):
    monkeypatch.setattr(
        rs,
        "_chunks",
        [
            {"path": "minzdrav_protocols/a/КП_1.pdf", "text": "a"},
            {"path": "minzdrav_protocols/a/КП_1.pdf", "text": "b"},
            {"path": "minzdrav_protocols/b/КП_2.pdf", "text": "c"},
        ],
    )
    allow = frozenset({"minzdrav_protocols/a/КП_1.pdf"})
    idx = rs._chunk_indices_for_path_allowlist(allow)
    assert idx == {0, 1}


def test_cap_lex_prefers_path_allowlist(monkeypatch):
    huge = set(range(20000))
    allow = frozenset({"minzdrav_protocols/x/КП.pdf"})
    monkeypatch.setattr(rs, "_chunks", [{"path": "minzdrav_protocols/x/КП.pdf", "text": "x"}])
    monkeypatch.setenv("RAG_LEX_MAX_CANDIDATES", "100")
    out = rs._cap_lex_candidate_indices(huge, path_allowlist_set=allow)
    assert out == {0}


def test_cap_lex_trims_without_full_sort(monkeypatch):
    huge = set(range(50000))
    monkeypatch.setenv("RAG_LEX_MAX_CANDIDATES", "100")
    monkeypatch.setattr(rs, "_lex_inverted_index", {})
    out = rs._cap_lex_candidate_indices(huge, path_allowlist_set=frozenset())
    assert out is not None
    assert len(out) == 100
