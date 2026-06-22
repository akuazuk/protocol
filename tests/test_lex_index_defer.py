"""Deferred lex inverted index on Render."""
from __future__ import annotations

import rag_server as rs


def test_ensure_lex_inverted_index_builds_once(monkeypatch) -> None:
    monkeypatch.setattr(rs, "_chunks", [{"path": "a/p.pdf", "text": "кашель бронхит", "title": "КП"}])
    monkeypatch.setattr(rs, "_lex_inverted_index", None)
    calls = {"n": 0}

    def fake_build(chunks):
        calls["n"] += 1
        return {"кашель": frozenset({0})}

    monkeypatch.setattr(rs, "_build_lex_inverted_index", fake_build)
    idx1 = rs._ensure_lex_inverted_index()
    idx2 = rs._ensure_lex_inverted_index()
    assert calls["n"] == 1
    assert idx1 is idx2
    assert "кашель" in idx1
