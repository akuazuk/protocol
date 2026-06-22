"""Render low-memory defaults (OOM mitigation)."""
from __future__ import annotations

import os

import rag_server


def test_apply_low_memory_defaults_sets_env(monkeypatch):
    for key in (
        "RAG_MEMORY_SAVER",
        "RAG_LEX_BM25_ALPHA",
        "RAG_LEXICAL_MAX_CHARS",
        "CONSULT_PREWARM_PROTOCOL_ICD_INDEX",
        "CONSULT_PREWARM_SUMMARY_ICD_INDEX",
        "CONSULT_REVIEW_CACHE_MAX",
        "PROTOCOL_SUMMARY_RAG_MERGE",
        "RAG_LEX_MAX_CANDIDATES",
        "RAG_LEX_MAX_UNION",
        "CONSULT_ALIGNMENT_ENABLED",
        "RAG_LEX_INDEX_DEFER",
        "CONSULT_CONCURRENCY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RENDER", "true")
    rag_server._apply_low_memory_defaults()
    assert os.environ.get("RAG_MEMORY_SAVER") == "1"
    assert os.environ.get("RAG_LEX_BM25_ALPHA") == "1.0"
    assert os.environ.get("RAG_LEXICAL_MAX_CHARS") == "4096"
    assert os.environ.get("PROTOCOL_SUMMARY_RAG_MERGE") == "0"
    assert os.environ.get("RAG_LEX_MAX_CANDIDATES") == "4000"
    assert os.environ.get("RAG_RETRIEVE_CONCURRENCY") == "1"
    assert os.environ.get("CONSULT_ALIGNMENT_ENABLED") == "0"
    assert os.environ.get("RAG_LEX_INDEX_DEFER") == "1"


def test_memory_saver_on_render(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("RAG_MEMORY_SAVER", raising=False)
    assert rag_server._memory_saver_enabled() is True


def test_memory_saver_explicit_off(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RAG_MEMORY_SAVER", "0")
    assert rag_server._memory_saver_enabled() is False
