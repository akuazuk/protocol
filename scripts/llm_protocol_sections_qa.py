#!/usr/bin/env python3
"""Document-level LLM QA: карта section_number → chunk_type."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_qa_prompt import SYSTEM_PROTOCOL_SECTIONS, build_protocol_sections_prompt
from clinical_knowledge.chunk_qa_schema import parse_protocol_sections_result

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
FALLBACK = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
OUT_DIR = ROOT / "data" / "ml" / "protocol_section_map"


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM protocol sections QA")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--doc-id", default="")
    args = parser.parse_args()

    if not _env_bool("CHUNK_QA_LLM"):
        print("SKIP: CHUNK_QA_LLM не включён", file=sys.stderr)
        return 0

    chunks_path = args.chunks or (DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK)
    if not chunks_path.is_file():
        print(f"Нет чанков: {chunks_path}", file=sys.stderr)
        return 1

    by_doc: dict[str, list[dict]] = defaultdict(list)
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            did = str(row.get("doc_id") or "")
            if args.doc_id and did != args.doc_id:
                continue
            by_doc[did].append(row)

    try:
        from rag_server import _extract_gemini_text, generate_gemini_consult_review_synthesize, get_gemini
        model = get_gemini()
    except Exception as e:
        print(f"SKIP: LLM недоступен ({e})", file=sys.stderr)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for did, chunks in sorted(by_doc.items()):
        if args.limit and n >= args.limit:
            break
        outline: dict[str, dict] = {}
        for ch in chunks:
            sn = str(ch.get("section_number") or ch.get("section_title") or "")
            if sn not in outline:
                outline[sn] = {
                    "section_number": ch.get("section_number") or "",
                    "section_title": ch.get("section_title") or "",
                    "page_from": ch.get("page_from"),
                }
        prompt = build_protocol_sections_prompt(
            doc_id=did,
            protocol_title=str(chunks[0].get("protocol_title") or ""),
            section_outline=list(outline.values())[:40],
        )
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        out_path = OUT_DIR / f"{did}.json"
        if out_path.is_file():
            n += 1
            continue
        try:
            resp = generate_gemini_consult_review_synthesize(
                model, SYSTEM_PROTOCOL_SECTIONS + "\n\n" + prompt, max_out=2000,
            )
            raw = _parse_json_obj(_extract_gemini_text(resp))
            parsed = parse_protocol_sections_result(raw or {})
            if parsed:
                out_path.write_text(json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            out_path.write_text(json.dumps({"doc_id": did, "error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1

    print(json.dumps({"processed": n, "out_dir": str(OUT_DIR)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
