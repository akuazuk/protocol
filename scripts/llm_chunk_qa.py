#!/usr/bin/env python3
"""Offline LLM-QA для rich-чанков (CHUNK_QA_LLM=1)."""
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

from clinical_knowledge.chunk_qa_prompt import SYSTEM_CHUNK_QA, build_chunk_qa_prompt
from clinical_knowledge.chunk_qa_schema import ChunkQaResult, parse_chunk_qa_result

DEFAULT_QUEUE = ROOT / "data" / "ml" / "chunk_qa_queue.jsonl"
DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.v2.jsonl"
FALLBACK_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "chunk_qa_fixes.jsonl"
CACHE_DIR = ROOT / "data" / "ml" / "chunk_qa_cache"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_chunks_index(path: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            if cid:
                idx[cid] = row
    return idx


def _run_llm(model: Any, prompt: str) -> str:
    from rag_server import _extract_gemini_text, generate_gemini_consult_review_synthesize
    resp = generate_gemini_consult_review_synthesize(model, SYSTEM_CHUNK_QA + "\n\n" + prompt, max_out=4000)
    return _extract_gemini_text(resp)


def process_batch(
    batch: list[dict[str, Any]],
    *,
    model: Any,
    protocol_title: str,
) -> list[ChunkQaResult]:
    prompt = build_chunk_qa_prompt(batch, protocol_title=protocol_title)
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache = _cache_path(key)
    if cache.is_file():
        raw_items = json.loads(cache.read_text(encoding="utf-8"))
    else:
        raw_text = _run_llm(model, prompt)
        raw_items = _parse_json_array(raw_text)
        cache.write_text(json.dumps(raw_items, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[ChunkQaResult] = []
    for item in raw_items:
        parsed = parse_chunk_qa_result(item)
        if parsed:
            results.append(parsed)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM chunk QA (offline)")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--doc-id", default="")
    args = parser.parse_args()

    if not _env_bool("CHUNK_QA_LLM", False):
        print("SKIP: CHUNK_QA_LLM не включён", file=sys.stderr)
        return 0

    chunks_path = args.chunks or (DEFAULT_CHUNKS if DEFAULT_CHUNKS.is_file() else FALLBACK_CHUNKS)
    if not chunks_path.is_file():
        print(f"Нет чанков: {chunks_path}", file=sys.stderr)
        return 1
    if not args.queue.is_file():
        print(f"Нет очереди: {args.queue}", file=sys.stderr)
        return 1

    try:
        from rag_server import get_gemini
        model = get_gemini()
    except Exception as e:
        print(f"SKIP: LLM недоступен ({e})", file=sys.stderr)
        return 0

    chunk_index = _load_chunks_index(chunks_path)
    queue_ids: list[str] = []
    with args.queue.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if args.doc_id and row.get("doc_id") != args.doc_id:
                continue
            queue_ids.append(str(row.get("chunk_id") or ""))
            if args.limit and len(queue_ids) >= args.limit:
                break

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cid in queue_ids:
        ch = chunk_index.get(cid)
        if ch:
            by_doc[str(ch.get("doc_id") or "")].append(ch)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_fixes = 0

    with args.out.open("w", encoding="utf-8") as out_fh:
        for doc_id, doc_chunks in by_doc.items():
            title = str(doc_chunks[0].get("protocol_title") or "") if doc_chunks else ""
            for i in range(0, len(doc_chunks), args.batch_size):
                batch = doc_chunks[i:i + args.batch_size]
                try:
                    results = process_batch(batch, model=model, protocol_title=title)
                except Exception as e:
                    for ch in batch:
                        out_fh.write(json.dumps({
                            "chunk_id": ch.get("chunk_id"),
                            "verdict": "ok",
                            "error": str(e),
                            "confidence": 0.0,
                        }, ensure_ascii=False) + "\n")
                    continue
                for res in results:
                    out_fh.write(json.dumps(res.model_dump(), ensure_ascii=False) + "\n")
                    n_fixes += 1

    print(json.dumps({"fixes_written": n_fixes, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
