"""Lazy retrieve: allowlist + chunk store без full corpus."""
from __future__ import annotations

from pathlib import Path

import pytest

import rag_server as rs
from clinical_knowledge.chunk_store import LazyChunkStore
from clinical_knowledge.corpus_path_manifest import CorpusPathManifest
from scripts.build_corpus_path_manifest import build_path_manifest, write_manifest


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chunks.mini.jsonl"


@pytest.fixture
def lazy_store(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunks.part.01.jsonl").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    entries = build_path_manifest(corpus)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(entries, manifest_path, corpus_dir=corpus)
    manifest = CorpusPathManifest.load(manifest_path)
    store = LazyChunkStore(manifest=manifest, corpus_dir=corpus)
    monkeypatch.setattr(rs, "_lazy_chunk_store", store)
    monkeypatch.setattr(rs, "_path_manifest", manifest)
    monkeypatch.setattr(rs, "_chunks", [])
    monkeypatch.setattr(rs, "_chunks_by_path", {})
    monkeypatch.setattr(rs, "_chunk_global_indices_by_path", {})
    monkeypatch.setenv("RAG_LAZY_RETRIEVE", "1")
    monkeypatch.setenv("RAG_LAZY_CHUNK_STORE", "1")
    monkeypatch.setenv("RAG_FORBID_FULL_CORPUS_RETRIEVE", "1")
    return store


def test_lazy_retrieve_with_allowlist(lazy_store):
    path = "minzdrav_protocols/_smoke/test_protocol_smoke.pdf"
    hits = rs._retrieve_core(
        "кашель бронхит",
        max_chunks=3,
        path_allowlist=[path],
    )
    assert hits
    assert hits[0].get("path") == path


def test_forbid_full_corpus_without_allowlist(lazy_store):
    hits = rs._retrieve_core("кашель бронхит", max_chunks=3)
    assert hits == []
