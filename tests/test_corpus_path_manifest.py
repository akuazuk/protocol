"""Tests for corpus path manifest builder and index."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.corpus_path_manifest import CorpusPathManifest
from scripts.build_corpus_path_manifest import build_path_manifest, write_manifest


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chunks.mini.jsonl"


def test_build_manifest_from_mini_fixture(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunks.part.01.jsonl").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    entries = build_path_manifest(corpus)
    assert len(entries) == 2
    paths = {e.path for e in entries}
    assert "minzdrav_protocols/_smoke/test_protocol_smoke.pdf" in paths
    out = tmp_path / "manifest.jsonl"
    write_manifest(entries, out, corpus_dir=corpus)
    loaded = CorpusPathManifest.load(out)
    stats = loaded.manifest_stats()
    assert stats["paths"] == 2
    assert stats["total_chunks"] == 2


def test_paths_by_icd():
    manifest = CorpusPathManifest()
    manifest.entries = {}
    from clinical_knowledge.corpus_path_manifest import PathManifestEntry

    manifest.entries["minzdrav_protocols/a/kp.pdf"] = PathManifestEntry(
        path="minzdrav_protocols/a/kp.pdf",
        rubric="a",
        chunk_count=1,
        icd10_codes=["J20.9"],
    )
    manifest._rebuild_indexes()
    found = manifest.paths_by_icd(["J20.9"])
    assert "minzdrav_protocols/a/kp.pdf" in found


def test_paths_by_rubric():
    manifest = CorpusPathManifest()
    from clinical_knowledge.corpus_path_manifest import PathManifestEntry

    manifest.entries["minzdrav_protocols/pulmo/x.pdf"] = PathManifestEntry(
        path="minzdrav_protocols/pulmo/x.pdf",
        rubric="pulmo",
        chunk_count=3,
    )
    manifest._rebuild_indexes()
    assert manifest.paths_by_rubric("pulmo") == ["minzdrav_protocols/pulmo/x.pdf"]
