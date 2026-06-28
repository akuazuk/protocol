#!/usr/bin/env python3
"""Добавить tags на существующие rich-чанки без повторного парсинга PDF."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import apply_indexable_flags
from clinical_knowledge.chunk_tags import build_chunk_tags, build_protocol_tags

CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
META_DIR = ROOT / "output" / "rich_meta"
OUT_TMP = CHUNKS.with_suffix(".tagged.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich rich chunks with tags")
    parser.add_argument("--dry-run", action="store_true", help="Stats only, do not overwrite")
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args()

    if not CHUNKS.is_file():
        print(f"Нет файла: {CHUNKS}", file=sys.stderr)
        return 1

    META_DIR.mkdir(parents=True, exist_ok=True)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    meta_by_doc: dict[str, dict] = {}
    signal_counts: Counter[str] = Counter()
    n = 0

    out_path = OUT_TMP if not args.dry_run else CHUNKS.with_suffix(".dryrun.jsonl")
    with out_path.open("w", encoding="utf-8") as out:
        with CHUNKS.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                pw = row.get("icd10_weights") or {}
                row["tags"] = build_chunk_tags(
                    text=str(row.get("text") or ""),
                    chunk_type=str(row.get("chunk_type") or "body"),
                    icd_codes=list(row.get("icd10_codes") or []),
                    care_setting=list(row.get("care_setting") or []),
                    protocol_weights=pw,
                    drugs=list(row.get("drugs") or []),
                    imaging=list(row.get("imaging") or []),
                    lab_tests=list(row.get("lab_tests") or []),
                )
                apply_indexable_flags(row)
                signal_counts[str((row.get("tags") or {}).get("signal") or "?")] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                did = str(row.get("doc_id") or "")
                if did:
                    by_doc[did].append(row)
                    if did not in meta_by_doc:
                        meta_by_doc[did] = {
                            "doc_id": did,
                            "source_path": row.get("source_path"),
                            "protocol_title": row.get("protocol_title"),
                            "protocol_kind": row.get("protocol_kind") or "",
                            "icd10_primary": row.get("icd10_codes") or [],
                        }
                n += 1
                if n % 10000 == 0:
                    print(f"  … {n} чанков", flush=True)

    stats = {"chunks": n, "signal": dict(signal_counts), "docs": len(by_doc)}
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        out_path.unlink(missing_ok=True)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    OUT_TMP.replace(CHUNKS)
    print(f"Обогащено чанков: {n}")

    for did, chunks in by_doc.items():
        counts = Counter(str(c.get("chunk_type") or "body") for c in chunks)
        base = meta_by_doc.get(did, {})
        sp = str(base.get("source_path") or chunks[0].get("source_path") or "")
        icd_codes: list[str] = []
        for c in chunks:
            icd_codes.extend(c.get("icd10_protocol") or c.get("icd10_codes") or [])
        icd_unique = list(dict.fromkeys(str(x).upper() for x in icd_codes if x))[:32]
        pt = build_protocol_tags(
            title=str(base.get("protocol_title") or ""),
            source_path=sp,
            protocol_kind=str(base.get("protocol_kind") or ""),
            icd_codes=icd_unique,
            chunk_type_counts=dict(counts),
        )
        meta_path = META_DIR / f"{did}.json"
        meta: dict = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["protocol_tags"] = pt
        meta["tags"] = pt
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Обновлено rich_meta: {len(by_doc)}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
