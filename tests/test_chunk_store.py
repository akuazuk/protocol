"""Tests for LazyChunkStore."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.chunk_store import LazyChunkStore
from clinical_knowledge.corpus_path_manifest import CorpusPathManifest
from scripts.build_corpus_path_manifest import build_path_manifest, write_manifest


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chunks.mini.jsonl"


def _make_store(tmp_path: Path) -> LazyChunkStore:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunks.part.01.jsonl").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    entries = build_path_manifest(corpus)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(entries, manifest_path, corpus_dir=corpus)
    manifest = CorpusPathManifest.load(manifest_path)
    return LazyChunkStore(
        manifest=manifest,
        corpus_dir=corpus,
        max_paths=2,
        max_chunks=4,
    )


def test_get_chunks_for_path(tmp_path: Path):
    store = _make_store(tmp_path)
    path = "minzdrav_protocols/_smoke/test_protocol_smoke.pdf"
    chunks = store.get_chunks_for_path(path)
    assert len(chunks) == 1
    assert "J20.9" in (chunks[0].get("text") or "")


def test_lru_eviction(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunks.part.01.jsonl").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    entries = build_path_manifest(corpus)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(entries, manifest_path, corpus_dir=corpus)
    manifest = CorpusPathManifest.load(manifest_path)
    store = LazyChunkStore(
        manifest=manifest,
        corpus_dir=corpus,
        max_paths=1,
        max_chunks=4,
    )
    paths = [
        "minzdrav_protocols/_smoke/test_protocol_smoke.pdf",
        "minzdrav_protocols/_smoke/general_ambulatory_mini.pdf",
    ]
    store.get_chunks_for_path(paths[0])
    store.get_chunks_for_path(paths[1])
    stats = store.cache_stats()
    assert stats["cached_paths"] == 1
    assert stats["evictions"] >= 1


def test_get_chunks_for_paths(tmp_path: Path):
    store = _make_store(tmp_path)
    rows = store.get_chunks_for_paths(
        [
            "minzdrav_protocols/_smoke/test_protocol_smoke.pdf",
            "minzdrav_protocols/_smoke/general_ambulatory_mini.pdf",
        ]
    )
    assert len(rows) == 2
