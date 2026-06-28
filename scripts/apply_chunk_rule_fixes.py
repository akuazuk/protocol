#!/usr/bin/env python3
"""Rule-based post-processing rich-чанков без LLM."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_entities import enrich_chunk_entities
from clinical_knowledge.chunk_quality import (
    apply_indexable_flags,
    fix_weak_section_title,
    strip_noise_lines,
    suggest_chunk_type,
)
from clinical_knowledge.chunk_tags import build_chunk_tags
from clinical_knowledge.chunk_type_infer import resolve_section_title

DEFAULT_IN = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
DEFAULT_OUT = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
DEFAULT_FIXES = ROOT / "data" / "ml" / "chunk_rule_fixes.jsonl"

_ICD_ENRICH_TYPES = frozenset({
    "diagnostics", "classification", "criteria_block", "treatment", "prevention",
})


def _extract_icd_from_text(text: str) -> list[str]:
    import re
    from icd_mkb import normalize_icd_code
    raw = re.findall(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b", text or "", re.I)
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        n = normalize_icd_code(c)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:28]


def rebuild_embedding_ready_text(chunk: dict[str, Any]) -> str:
    section_title = str(chunk.get("section_title") or "")
    chunk_text = str(chunk.get("text") or "")
    icd_codes = list(chunk.get("icd10_codes") or [])
    populations = list(chunk.get("population") or [])
    ctype = str(chunk.get("chunk_type") or "body")
    emb = section_title + "\n" + chunk_text
    if icd_codes:
        emb = "МКБ-10: " + ", ".join(icd_codes[:12]) + "\n" + emb
    elif chunk.get("icd10_protocol") and ctype == "protocol_overview":
        proto = list(chunk.get("icd10_protocol") or [])[:12]
        if proto:
            emb = "МКБ-10 (протокол): " + ", ".join(proto) + "\n" + emb
    if populations:
        emb = "Популяция: " + ", ".join(populations) + "\n" + emb
    return emb


def apply_rules_to_chunk(
    chunk: dict[str, Any],
    *,
    fixes: list[dict[str, Any]],
) -> dict[str, Any]:
    ch = dict(chunk)
    cid = str(ch.get("chunk_id") or "")

    # icd10_protocol from current icd list if missing
    if not ch.get("icd10_protocol"):
        ch["icd10_protocol"] = list(ch.get("icd10_codes") or [])

    text = str(ch.get("text") or "")
    cleaned = strip_noise_lines(text)
    if cleaned != text:
        fixes.append({"chunk_id": cid, "op": "trim_text", "before_len": len(text), "after_len": len(cleaned)})
        ch["text"] = cleaned

    new_title = fix_weak_section_title(ch)
    if new_title != ch.get("section_title"):
        fixes.append({
            "chunk_id": cid,
            "op": "fix_section_title",
            "before": ch.get("section_title"),
            "after": new_title,
        })
        ch["section_title"] = new_title

    suggested = suggest_chunk_type(ch)
    if suggested:
        fixes.append({
            "chunk_id": cid,
            "op": "set_chunk_type",
            "before": ch.get("chunk_type"),
            "after": suggested,
        })
        ch["chunk_type"] = suggested

    text_icd = _extract_icd_from_text(str(ch.get("text") or ""))
    old_icd = list(ch.get("icd10_codes") or [])
    ctype = str(ch.get("chunk_type") or "body")
    if ctype in _ICD_ENRICH_TYPES or text_icd:
        if text_icd != old_icd:
            fixes.append({
                "chunk_id": cid,
                "op": "set_icd10_text_only",
                "before_count": len(old_icd),
                "after_count": len(text_icd),
            })
            ch["icd10_codes"] = text_icd if text_icd else old_icd[:6]
    elif len(old_icd) > 12:
        ch["icd10_codes"] = old_icd[:12]

    enrich_chunk_entities(ch)

    pw = ch.get("icd10_weights") or {}
    ch["tags"] = build_chunk_tags(
        text=str(ch.get("text") or ""),
        chunk_type=str(ch.get("chunk_type") or "body"),
        icd_codes=list(ch.get("icd10_codes") or []),
        care_setting=list(ch.get("care_setting") or []),
        protocol_weights=pw,
        drugs=list(ch.get("drugs") or []),
        imaging=list(ch.get("imaging") or []),
        lab_tests=list(ch.get("lab_tests") or []),
    )

    apply_indexable_flags(ch)
    if ch.get("indexable") is False:
        fixes.append({"chunk_id": cid, "op": "set_indexable_false", "noise_flags": ch.get("noise_flags")})

    ch["embedding_ready_text"] = rebuild_embedding_ready_text(ch)
    ch["quality_score"] = ch.get("quality_score") or 0.0
    return ch


def merge_short_chunks(chunks: list[dict[str, Any]], fixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Склеить соседние короткие чанки одного раздела."""
    if not chunks:
        return chunks
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        text = (ch.get("text") or "").strip()
        if (
            len(text) < 80
            and i + 1 < len(chunks)
            and chunks[i + 1].get("doc_id") == ch.get("doc_id")
            and chunks[i + 1].get("section_number") == ch.get("section_number")
            and (chunks[i + 1].get("text") or "").strip()
        ):
            nxt = chunks[i + 1]
            merged_text = text + "\n" + (nxt.get("text") or "").strip()
            merged = dict(ch)
            merged["text"] = merged_text
            merged["chunk_id"] = str(ch.get("chunk_id")) + "_m0"
            merged["page_to"] = nxt.get("page_to") or ch.get("page_to")
            fixes.append({
                "chunk_id": merged["chunk_id"],
                "op": "merge_short_chunks",
                "merged_from": [ch.get("chunk_id"), nxt.get("chunk_id")],
            })
            merged = apply_rules_to_chunk(merged, fixes=fixes)
            out.append(merged)
            i += 2
            continue
        out.append(ch)
        i += 1
    return out


def process_file(
    in_path: Path,
    out_path: Path,
    fixes_path: Path,
    *,
    merge_short: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    fixes: list[dict[str, Any]] = []
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n = 0

    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit and n >= limit:
                break
            row = json.loads(line)
            n += 1
            did = str(row.get("doc_id") or "")
            fixed = apply_rules_to_chunk(row, fixes=fixes)
            by_doc[did].append(fixed)

    out_chunks: list[dict[str, Any]] = []
    for did in sorted(by_doc.keys()):
        doc_chunks = by_doc[did]
        if merge_short:
            doc_chunks = merge_short_chunks(doc_chunks, fixes)
        out_chunks.extend(doc_chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for ch in out_chunks:
            out.write(json.dumps(ch, ensure_ascii=False) + "\n")

    fixes_path.parent.mkdir(parents=True, exist_ok=True)
    with fixes_path.open("w", encoding="utf-8") as ff:
        for fx in fixes:
            ff.write(json.dumps(fx, ensure_ascii=False) + "\n")

    return {
        "input_chunks": n,
        "output_chunks": len(out_chunks),
        "fixes": len(fixes),
        "indexable_false": sum(1 for c in out_chunks if c.get("indexable") is False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply rule-based chunk fixes")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.in_path.is_file():
        print(f"Нет файла: {args.in_path}", file=sys.stderr)
        return 1

    stats = process_file(
        args.in_path,
        args.out_path,
        args.fixes,
        merge_short=not args.no_merge,
        limit=args.limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
