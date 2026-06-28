#!/usr/bin/env python3
"""Применить LLM-fixes к rich-чанкам."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import apply_indexable_flags
from clinical_knowledge.chunk_tags import build_chunk_tags

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
FALLBACK_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
DEFAULT_FIXES = ROOT / "data" / "ml" / "chunk_qa_fixes.jsonl"
DEFAULT_OUT = ROOT / "output" / "rich_chunks" / "rich_chunks.final.jsonl"
DEFAULT_REVIEW = ROOT / "data" / "ml" / "chunk_qa_review.jsonl"

AUTO_CONFIDENCE = 0.85
REVIEW_CONFIDENCE = 0.6


def _rebuild_embedding(ch: dict[str, Any]) -> None:
    section_title = str(ch.get("section_title") or "")
    chunk_text = str(ch.get("text") or "")
    icd_codes = list(ch.get("icd10_codes") or [])
    populations = list(ch.get("population") or [])
    emb = section_title + "\n" + chunk_text
    if icd_codes:
        emb = "МКБ-10: " + ", ".join(icd_codes[:12]) + "\n" + emb
    if populations:
        emb = "Популяция: " + ", ".join(populations) + "\n" + emb
    ch["embedding_ready_text"] = emb


def _text_diff_ratio(before: str, after: str) -> float:
    if not before:
        return 1.0 if after else 0.0
    return abs(len(after) - len(before)) / max(len(before), 1)


def apply_fix(ch: dict[str, Any], fix: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    """Вернуть (chunk, applied, disposition). disposition: applied|review|skipped"""
    conf = float(fix.get("confidence") or 0.0)
    verdict = str(fix.get("verdict") or "ok")

    if verdict == "ok" or conf < REVIEW_CONFIDENCE:
        return ch, False, "skipped"

    if conf < AUTO_CONFIDENCE:
        return ch, False, "review"

    out = dict(ch)
    if verdict == "drop":
        out["indexable"] = False
        nf = list(out.get("noise_flags") or [])
        if "llm_drop" not in nf:
            nf.append("llm_drop")
        out["noise_flags"] = nf
        return out, True, "applied"

    if fix.get("corrected_chunk_type"):
        out["chunk_type"] = fix["corrected_chunk_type"]
    if fix.get("corrected_section_title"):
        out["section_title"] = fix["corrected_section_title"]
    if fix.get("clean_text"):
        before = str(ch.get("text") or "")
        after = str(fix["clean_text"])
        if _text_diff_ratio(before, after) <= 0.35:
            out["text"] = after

    entities = fix.get("entities") or {}
    if entities.get("exam"):
        out["lab_tests"] = list(dict.fromkeys(list(out.get("lab_tests") or []) + entities["exam"]))[:30]
    if entities.get("drug"):
        out["drugs"] = list(dict.fromkeys(list(out.get("drugs") or []) + entities["drug"]))[:30]

    tags = build_chunk_tags(
        text=str(out.get("text") or ""),
        chunk_type=str(out.get("chunk_type") or "body"),
        icd_codes=list(out.get("icd10_codes") or []),
        care_setting=list(out.get("care_setting") or []),
        protocol_weights=out.get("icd10_weights") or {},
        drugs=list(out.get("drugs") or []),
        imaging=list(out.get("imaging") or []),
        lab_tests=list(out.get("lab_tests") or []),
    )
    if fix.get("obligation"):
        tags["obligation"] = fix["obligation"]
    out["tags"] = tags
    apply_indexable_flags(out)
    _rebuild_embedding(out)
    return out, True, "applied"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LLM chunk QA fixes")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    chunks_path = args.chunks or (DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK_CHUNKS)
    if not chunks_path.is_file():
        print(f"Нет чанков: {chunks_path}", file=sys.stderr)
        return 1
    if not args.fixes.is_file():
        print(f"Нет fixes (копируем chunks as-is): {args.fixes}", file=sys.stderr)
        args.out.write_bytes(chunks_path.read_bytes())
        return 0

    fixes_by_id: dict[str, dict[str, Any]] = {}
    with args.fixes.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            if cid:
                fixes_by_id[cid] = row

    applied = review = skipped = 0
    args.review.parent.mkdir(parents=True, exist_ok=True)
    review_fh = args.review.open("w", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with chunks_path.open(encoding="utf-8") as inp, args.out.open("w", encoding="utf-8") as outp:
        for line in inp:
            ch = json.loads(line)
            cid = str(ch.get("chunk_id") or "")
            fix = fixes_by_id.get(cid)
            if fix:
                ch, ok, disp = apply_fix(ch, fix)
                if disp == "applied":
                    applied += 1
                elif disp == "review":
                    review += 1
                    review_fh.write(json.dumps({"chunk_id": cid, "fix": fix}, ensure_ascii=False) + "\n")
                else:
                    skipped += 1
            outp.write(json.dumps(ch, ensure_ascii=False) + "\n")

    review_fh.close()
    print(json.dumps({"applied": applied, "review": review, "skipped": skipped}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
