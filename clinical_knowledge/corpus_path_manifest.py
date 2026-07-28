"""Path-level manifest: rubric, ICD, chunk counts - без загрузки текстов в RAM."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _norm_path(p: str) -> str:
    return str(p or "").replace("\\", "/").strip()


def _rubric_from_path(path: str) -> str:
    parts = _norm_path(path).split("/")
    if len(parts) >= 2 and parts[0] == "minzdrav_protocols":
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return ""


@dataclass
class PathManifestEntry:
    path: str
    rubric: str = ""
    chunk_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    icd10_codes: list[str] = field(default_factory=list)
    chunk_types: dict[str, int] = field(default_factory=dict)
    population: list[str] = field(default_factory=list)
    source_part: str = ""
    byte_offsets: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rubric": self.rubric,
            "chunk_count": self.chunk_count,
            "chunk_ids": self.chunk_ids,
            "icd10_codes": self.icd10_codes,
            "chunk_types": self.chunk_types,
            "population": self.population,
            "source_part": self.source_part,
            "byte_offsets": self.byte_offsets,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> PathManifestEntry:
        path = _norm_path(str(row.get("path") or ""))
        icd = row.get("icd10_codes") or []
        pops = row.get("population") or []
        types = row.get("chunk_types") or {}
        offsets = row.get("byte_offsets") or []
        return cls(
            path=path,
            rubric=str(row.get("rubric") or _rubric_from_path(path)),
            chunk_count=int(row.get("chunk_count") or 0),
            chunk_ids=[str(x) for x in (row.get("chunk_ids") or []) if x],
            icd10_codes=[str(x).upper() for x in icd if x],
            chunk_types={str(k): int(v) for k, v in types.items() if k},
            population=[str(x) for x in pops if x],
            source_part=str(row.get("source_part") or ""),
            byte_offsets=[
                [int(a), int(b)] for a, b in offsets if isinstance(a, (int, float)) and isinstance(b, (int, float))
            ],
        )


class CorpusPathManifest:
    """In-memory index по path manifest JSONL."""

    def __init__(self) -> None:
        self.entries: dict[str, PathManifestEntry] = {}
        self._icd_index: dict[str, list[str]] = {}
        self._rubric_index: dict[str, list[str]] = {}
        self.manifest_sha256: str = ""
        self.generated_at: str = ""

    @classmethod
    def load(cls, path: Path) -> CorpusPathManifest:
        inst = cls()
        if not path.is_file():
            return inst
        inst.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as f:
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
                if row.get("_header"):
                    inst.generated_at = str(row.get("generated_at") or "")
                    continue
                entry = PathManifestEntry.from_dict(row)
                if not entry.path:
                    continue
                inst.entries[entry.path] = entry
        inst._rebuild_indexes()
        return inst

    def _rebuild_indexes(self) -> None:
        self._icd_index = {}
        self._rubric_index = {}
        for path, entry in self.entries.items():
            rub = entry.rubric or _rubric_from_path(path)
            if rub:
                self._rubric_index.setdefault(rub, []).append(path)
            for code in entry.icd10_codes:
                c = str(code).upper().strip()
                if c:
                    self._icd_index.setdefault(c, []).append(path)
            for code in entry.icd10_codes:
                c = str(code).upper().strip()
                if "." in c:
                    base = c.split(".", 1)[0]
                    self._icd_index.setdefault(base, []).append(path)

    def paths_by_icd(self, codes: list[str], *, limit: int = 15) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in codes or []:
            c = str(raw).upper().strip()
            if not c:
                continue
            for key in (c, c.split(".", 1)[0] if "." in c else ""):
                if not key:
                    continue
                for p in self._icd_index.get(key, ()):
                    if p in seen:
                        continue
                    seen.add(p)
                    out.append(p)
                    if len(out) >= limit:
                        return out
        return out[:limit]

    def paths_by_rubric(self, rubric: str, *, limit: int = 50) -> list[str]:
        rub = str(rubric or "").strip()
        if not rub:
            return []
        return list(self._rubric_index.get(rub, ()))[:limit]

    def get(self, path: str) -> PathManifestEntry | None:
        norm = _norm_path(path)
        if norm in self.entries:
            return self.entries[norm]
        for p, entry in self.entries.items():
            if p.endswith(norm.split("/")[-1]) or norm.endswith(p.split("/")[-1]):
                return entry
        return None

    def manifest_stats(self) -> dict[str, Any]:
        total_chunks = sum(e.chunk_count for e in self.entries.values())
        rubrics = len(self._rubric_index)
        icd_keys = len(self._icd_index)
        return {
            "paths": len(self.entries),
            "total_chunks": total_chunks,
            "rubrics": rubrics,
            "icd_index_keys": icd_keys,
            "manifest_sha256": self.manifest_sha256,
            "generated_at": self.generated_at,
        }
