#!/usr/bin/env python3
"""Path-level manifest: один JSONL-ряд на PDF (rubric, ICD, offsets)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.corpus_path_manifest import PathManifestEntry, _norm_path, _rubric_from_path


def _collect_parts(corpus_dir: Path) -> list[Path]:
    parts = sorted(corpus_dir.glob("chunks.part.*.jsonl"))
    if not parts:
        parts = sorted(corpus_dir.glob("*.jsonl"))
    return parts


def build_path_manifest(corpus_dir: Path) -> list[PathManifestEntry]:
    """Один проход по JSONL: агрегация по source_path."""
    accum: dict[str, dict] = defaultdict(
        lambda: {
            "chunk_ids": [],
            "icd10_codes": set(),
            "chunk_types": defaultdict(int),
            "population": set(),
            "source_part": "",
            "byte_offsets": [],
        }
    )
    parts = _collect_parts(corpus_dir)
    if not parts:
        return []
    for part in parts:
        part_name = part.name
        with part.open("rb") as fb:
            offset = 0
            while True:
                start = offset
                line_b = fb.readline()
                if not line_b:
                    break
                offset += len(line_b)
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                p = _norm_path(str(row.get("source_path") or ""))
                if not p:
                    continue
                bucket = accum[p]
                if not bucket["source_part"]:
                    bucket["source_part"] = part_name
                cid = str(row.get("chunk_id") or "").strip()
                if cid:
                    bucket["chunk_ids"].append(cid)
                ctype = str(row.get("chunk_type") or "body").strip() or "body"
                bucket["chunk_types"][ctype] += 1
                icd = row.get("icd10_codes")
                if isinstance(icd, list):
                    for c in icd:
                        if c:
                            bucket["icd10_codes"].add(str(c).upper())
                pops = row.get("population")
                if isinstance(pops, list):
                    for x in pops:
                        if x:
                            bucket["population"].add(str(x))
                bucket["byte_offsets"].append([start, offset])
    entries: list[PathManifestEntry] = []
    for path in sorted(accum.keys()):
        b = accum[path]
        entries.append(
            PathManifestEntry(
                path=path,
                rubric=_rubric_from_path(path),
                chunk_count=len(b["chunk_ids"]),
                chunk_ids=b["chunk_ids"],
                icd10_codes=sorted(b["icd10_codes"]),
                chunk_types=dict(b["chunk_types"]),
                population=sorted(b["population"]),
                source_part=b["source_part"],
                byte_offsets=b["byte_offsets"],
            )
        )
    return entries


def write_manifest(entries: list[PathManifestEntry], output: Path, *, corpus_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "_header": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "paths_count": len(entries),
        "total_chunks": sum(e.chunk_count for e in entries),
    }
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus_path_manifest.jsonl")
    parser.add_argument("--corpus", default="corpus_chunks_parts", help="Dir with chunks.part.*.jsonl")
    parser.add_argument(
        "--output",
        default="data/catalog/corpus_path_manifest.jsonl",
        help="Output manifest JSONL",
    )
    args = parser.parse_args()
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = (ROOT / corpus).resolve()
    if not corpus.is_dir():
        print(f"Нет каталога {corpus}", file=sys.stderr)
        return 1
    entries = build_path_manifest(corpus)
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    write_manifest(entries, out, corpus_dir=corpus)
    stats = {
        "paths": len(entries),
        "total_chunks": sum(e.chunk_count for e in entries),
        "output": str(out),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
