"""Lazy chunk store: чтение slim-чанков с диска по path + LRU cache."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from clinical_knowledge.corpus_path_manifest import CorpusPathManifest, PathManifestEntry, _norm_path
from clinical_knowledge.lazy_rag_config import (
    chunk_cache_max_chunks,
    chunk_cache_paths,
    chunks_data_root,
    manifest_path,
)


def _memory_saver_enabled() -> bool:
    import os

    v = (os.environ.get("RAG_MEMORY_SAVER") or "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    if (os.environ.get("RAG_CHUNKS_DIR") or "").strip():
        return True
    return False


def _jsonl_row_to_slim(
    row: dict[str, Any],
    *,
    path: str,
    chunk_index: int,
    include_embedding: bool = False,
    include_entities: bool = False,
) -> dict[str, Any]:
    """Формат retrieve/gather - совместим с rag_server._load_chunks_from_jsonl."""
    text = (row.get("text") or "").strip()
    rec: dict[str, Any] = {
        "path": path,
        "text": text,
        "title": "",
        "category": "",
        "kind": (row.get("chunk_type") or "body").strip() or "body",
        "chunk_index": chunk_index,
        "chunk_id": row.get("chunk_id"),
    }
    if row.get("doc_id") or row.get("protocol_title"):
        rec["rich_chunk"] = True
        spec = (row.get("specialty_slug") or "").strip()
        if spec:
            rec["category"] = spec
        pops = row.get("population")
        if isinstance(pops, list) and pops:
            rec["chunk_population"] = [str(x) for x in pops][:8]
    for fld in ("section_path", "section_title", "point_numbers", "icd10_codes", "icd10_weights"):
        if fld in row and row[fld]:
            rec[fld] = row[fld]
    if row.get("page_from"):
        rec["page_from"] = int(row.get("page_from") or 0)
    if row.get("page_to"):
        rec["page_to"] = int(row.get("page_to") or 0)
    if row.get("page_to"):
        rec["page_to"] = int(row.get("page_to") or 0)
    keep_emb = include_embedding or not _memory_saver_enabled()
    if keep_emb:
        ert = (row.get("embedding_ready_text") or "").strip()
        if ert and ert != text:
            rec["lex_text"] = ert
            rec["embedding_ready_text"] = ert
        emb = row.get("embedding")
        if isinstance(emb, list) and len(emb) >= 8:
            rec["embedding"] = [float(x) for x in emb]
    elif text:
        import os

        lex_cap = int(os.environ.get("RAG_LEXICAL_MAX_CHARS", "0") or "0")
        if lex_cap > 0 and len(text) > lex_cap:
            rec["lex_text"] = text[:lex_cap]
    if include_entities:
        rec["chunk_type"] = (row.get("chunk_type") or rec.get("kind") or "body").strip() or "body"
        for fld in ("drugs", "imaging", "lab_tests", "procedures", "dosages"):
            vals = row.get(fld)
            if isinstance(vals, list) and vals:
                rec[fld] = vals
    return rec


class LazyChunkStore:
    """Disk-backed chunk access with LRU cache by PDF path."""

    def __init__(
        self,
        *,
        manifest: CorpusPathManifest,
        corpus_dir: Path,
        max_paths: int | None = None,
        max_chunks: int | None = None,
        protocols_by_path: dict[str, dict] | None = None,
        protocol_meta: dict[str, dict] | None = None,
    ) -> None:
        self.manifest = manifest
        self.corpus_dir = corpus_dir
        self.max_paths = max_paths if max_paths is not None else chunk_cache_paths()
        self.max_chunks = max_chunks if max_chunks is not None else chunk_cache_max_chunks()
        self.protocols_by_path = protocols_by_path or {}
        self.protocol_meta = protocol_meta or {}
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._cache_chunk_count = 0
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "disk_reads": 0}

    @classmethod
    def from_env(
        cls,
        *,
        protocols_by_path: dict[str, dict] | None = None,
        protocol_meta: dict[str, dict] | None = None,
    ) -> LazyChunkStore | None:
        mpath = manifest_path()
        if not mpath.is_file():
            return None
        manifest = CorpusPathManifest.load(mpath)
        if not manifest.entries:
            return None
        corpus = chunks_data_root()
        # Каталог с JSONL-чанками может лежать в corpus_chunks_parts (part-файлы) или
        # в output/rich_chunks (rich_chunks.jsonl на Render persistent disk). Берём первый
        # каталог, где реально есть *.jsonl с чанками, иначе fallback на сам corpus.
        corpus_dir = None
        for cand in (corpus / "corpus_chunks_parts", corpus / "output" / "rich_chunks", corpus):
            try:
                if cand.is_dir() and any(cand.glob("*.jsonl")):
                    corpus_dir = cand
                    break
            except OSError:
                continue
        if corpus_dir is None:
            if corpus.is_dir():
                corpus_dir = corpus
            else:
                return None
        return cls(
            manifest=manifest,
            corpus_dir=corpus_dir,
            protocols_by_path=protocols_by_path,
            protocol_meta=protocol_meta,
        )

    def cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "cached_paths": len(self._cache),
                "cached_chunks": self._cache_chunk_count,
                "max_paths": self.max_paths,
                "max_chunks": self.max_chunks,
            }

    def _enrich_chunk(self, ch: dict[str, Any]) -> None:
        p = str(ch.get("path") or "")
        pr = self.protocols_by_path.get(p) or {}
        pm = self.protocol_meta.get(p) or {}
        if not (ch.get("title") or "").strip():
            ch["title"] = (pr.get("title") or pm.get("title") or "").strip()
        if not (ch.get("category") or "").strip():
            ch["category"] = (pr.get("category") or pm.get("category") or "").strip()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_paths or self._cache_chunk_count > self.max_chunks:
            if not self._cache:
                break
            _, evicted = self._cache.popitem(last=False)
            self._cache_chunk_count -= len(evicted)
            self._stats["evictions"] += 1

    def _put_cache(self, path: str, chunks: list[dict]) -> list[dict]:
        with self._lock:
            if path in self._cache:
                old = self._cache.pop(path)
                self._cache_chunk_count -= len(old)
            self._cache[path] = chunks
            self._cache_chunk_count += len(chunks)
            self._cache.move_to_end(path)
            self._evict_if_needed()
        return chunks

    def _read_offsets(self, part_file: Path, offsets: list[list[int]]) -> list[dict]:
        rows: list[dict] = []
        with part_file.open("rb") as f:
            for pair in offsets:
                if len(pair) < 2:
                    continue
                start, end = int(pair[0]), int(pair[1])
                f.seek(start)
                raw = f.read(max(0, end - start))
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _scan_part_for_path(self, part_file: Path, path: str) -> list[dict]:
        norm = _norm_path(path)
        rows: list[dict] = []
        with part_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                sp = _norm_path(str(row.get("source_path") or ""))
                if sp == norm or sp.endswith(norm.split("/")[-1]):
                    rows.append(row)
        return rows

    def _rows_match_entry(self, rows: list[dict], entry: PathManifestEntry) -> bool:
        """Проверка, что offset-чтение попало в нужный протокол (защита от устаревшего манифеста)."""
        norm = _norm_path(entry.path)
        tail = norm.split("/")[-1]
        for row in rows:
            sp = _norm_path(str(row.get("source_path") or ""))
            if sp:
                return sp == norm or sp.endswith(tail) or norm.endswith(sp.split("/")[-1])
        return False

    def _load_rows_for_entry(self, entry: PathManifestEntry, *, semantic: bool = False) -> list[dict]:
        self._stats["disk_reads"] += 1
        part_name = entry.source_part
        part_file = self.corpus_dir / part_name if part_name else None
        raw_rows: list[dict] = []
        if part_file and part_file.is_file() and entry.byte_offsets:
            raw_rows = self._read_offsets(part_file, entry.byte_offsets)
            if raw_rows and not self._rows_match_entry(raw_rows, entry):
                raw_rows = []
        if not raw_rows and part_file and part_file.is_file():
            raw_rows = self._scan_part_for_path(part_file, entry.path)
        if not raw_rows:
            for part in sorted(self.corpus_dir.glob("*.jsonl")):
                raw_rows = self._scan_part_for_path(part, entry.path)
                if raw_rows:
                    break
        out: list[dict] = []
        for i, row in enumerate(raw_rows):
            ch = _jsonl_row_to_slim(
                row,
                path=entry.path,
                chunk_index=i,
                include_embedding=semantic,
                include_entities=semantic,
            )
            self._enrich_chunk(ch)
            out.append(ch)
        out.sort(key=lambda r: (r.get("page_from") or 0, r.get("chunk_index") or 0))
        for i, ch in enumerate(out):
            ch["chunk_index"] = i
        return out

    def get_chunks_for_path(
        self,
        path: str,
        *,
        max_chunks: int = 64,
        chunk_types: set[str] | None = None,
        semantic: bool = False,
    ) -> list[dict]:
        norm = _norm_path(path)
        if not norm:
            return []
        cache_key = f"{norm}:semantic" if semantic else norm
        with self._lock:
            if cache_key in self._cache:
                self._stats["hits"] += 1
                self._cache.move_to_end(cache_key)
                cached = list(self._cache[cache_key])
            else:
                cached = None
        if cached is None:
            self._stats["misses"] += 1
            entry = self.manifest.get(norm)
            if entry is None:
                return []
            loaded = self._load_rows_for_entry(entry, semantic=semantic)
            cached = self._put_cache(cache_key, loaded)
        if chunk_types:
            cached = [c for c in cached if str(c.get("kind") or "body") in chunk_types]
        return cached[: max(1, max_chunks)]

    def get_chunks_for_paths(
        self,
        paths: list[str],
        *,
        max_chunks_per_path: int = 64,
        max_total: int = 512,
        chunk_types: set[str] | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        seen_paths: set[str] = set()
        for p in paths or []:
            norm = _norm_path(p)
            if not norm or norm in seen_paths:
                continue
            seen_paths.add(norm)
            rows = self.get_chunks_for_path(
                norm, max_chunks=max_chunks_per_path, chunk_types=chunk_types
            )
            out.extend(rows)
            if len(out) >= max_total:
                return out[:max_total]
        return out
