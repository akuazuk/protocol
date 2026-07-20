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
    monkeypatch.delenv("RENDER_PLAN", raising=False)
    monkeypatch.setenv("RENDER", "true")
    rag_server._apply_low_memory_defaults()
    assert os.environ.get("RAG_MEMORY_SAVER") == "1"
    assert os.environ.get("RAG_LEX_BM25_ALPHA") == "1.0"
    assert os.environ.get("RAG_LEXICAL_MAX_CHARS") == "4096"
    assert os.environ.get("PROTOCOL_SUMMARY_RAG_MERGE") == "1"
    assert os.environ.get("RAG_LEX_MAX_CANDIDATES") == "4000"
    assert os.environ.get("RAG_RETRIEVE_CONCURRENCY") == "1"
    assert os.environ.get("CONSULT_ALIGNMENT_ENABLED") == "0"
    assert os.environ.get("RAG_LEX_INDEX_DEFER") == "1"


def test_memory_saver_on_render(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("RAG_MEMORY_SAVER", raising=False)
    assert rag_server._memory_saver_enabled() is True


def test_render_extended_ram_standard_plan(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_PLAN", "standard")
    assert rag_server._render_extended_ram() is True


def test_apply_standard_plan_defaults(monkeypatch):
    for key in (
        "RAG_LEX_MAX_CANDIDATES",
        "CONSULT_ALIGNMENT_ENABLED",
        "CONSULT_RENDER_L2_SKIP_LLM",
        "RAG_LEX_INDEX_DEFER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_PLAN", "standard")
    rag_server._apply_low_memory_defaults()
    assert os.environ.get("RAG_LEX_MAX_CANDIDATES") == "8000"
    assert os.environ.get("CONSULT_ALIGNMENT_ENABLED") == "1"
    assert os.environ.get("CONSULT_RENDER_L2_SKIP_LLM") == "0"
    assert os.environ.get("RAG_LEX_INDEX_DEFER") == "1"
    assert os.environ.get("CONSULT_CONCURRENCY") == "1"
    assert os.environ.get("CONSULT_REVIEW_FORBID_FULL_CORPUS") == "1"


def test_render_high_ram_by_ram_mb(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_RAM_MB", "4096")
    monkeypatch.delenv("RENDER_PLAN", raising=False)
    assert rag_server._render_high_ram() is True


def test_render_high_ram_standard_is_not_high(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_PLAN", "standard")
    monkeypatch.delenv("RENDER_RAM_MB", raising=False)
    assert rag_server._render_high_ram() is False
    assert rag_server._render_extended_ram() is True


def test_apply_high_ram_defaults(monkeypatch):
    for key in (
        "RAG_GEMINI_EMBED_RERANK",
        "RAG_LEX_BM25_ALPHA",
        "RAG_EMBED_POOL_MERGE",
        "RAG_VECTOR_INDEX",
        "RAG_STARTUP_MODE",
        "RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER",
        "PROTOCOL_SUMMARY_RAG_MERGE",
        "RAG_MEMORY_SAVER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_RAM_MB", "4096")
    monkeypatch.delenv("RENDER_PLAN", raising=False)
    rag_server._apply_low_memory_defaults()
    assert os.environ.get("RAG_GEMINI_EMBED_RERANK") == "1"
    assert os.environ.get("RAG_LEX_BM25_ALPHA") == "0.55"
    assert os.environ.get("RAG_EMBED_POOL_MERGE") == "1"
    assert os.environ.get("RAG_VECTOR_INDEX") == "1"
    assert os.environ.get("RAG_STARTUP_MODE") == "full"
    assert os.environ.get("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER") == "0"
    assert os.environ.get("PROTOCOL_SUMMARY_RAG_MERGE") == "1"
    assert os.environ.get("RAG_MEMORY_SAVER") == "0"


def test_memory_saver_explicit_off(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RAG_MEMORY_SAVER", "0")
    assert rag_server._memory_saver_enabled() is False
