#!/usr/bin/env python3
"""Кросс-карта rich_chunks chunk_id → global index (для Render без corpus_chunks_parts в RAM)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _norm_path(p: str) -> str:
    return str(p or "").replace("\\", "/").strip()


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().replace("ё", "е")).strip()


def _jsonl_parts(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(inp.glob("chunks.part.*.jsonl")) or sorted(inp.glob("*.jsonl"))


def _load_corpus_by_path(parts: list[Path]) -> dict[str, list[dict]]:
    import rag_server as rs

    chunks = rs._load_chunks_from_jsonl(parts)
    by_path: dict[str, list[dict]] = {}
    for i, ch in enumerate(chunks):
        p = _norm_path(str(ch.get("path") or ""))
        if not p:
            continue
        row = dict(ch)
        row["_global_index"] = i
        by_path.setdefault(p, []).append(row)
    return by_path


def _load_rich_rows(rich_path: Path) -> list[dict]:
    rows: list[dict] = []
    with rich_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _pick_match(rich: dict, corpus_rows: list[dict]) -> int | None:
    rt = _norm_text(rich.get("text") or "")
    if len(rt) < 24:
        return None
    rpage = int(rich.get("page_from") or rich.get("page_to") or 0)
    rtype = str(rich.get("chunk_type") or "body").strip()
    best: tuple[float, int] | None = None
    for ch in corpus_rows:
        ct = _norm_text(ch.get("text") or "")
        if len(ct) < 20:
            continue
        score = 0.0
        cpage = int(ch.get("page_from") or ch.get("page_to") or 0)
        ctype = str(ch.get("chunk_type") or ch.get("kind") or "body").strip()
        if rpage and cpage and rpage == cpage:
            score += 3.0
        if rtype == ctype:
            score += 1.5
        if rt[:80] == ct[:80]:
            score += 8.0
        elif rt[:48] in ct or ct[:48] in rt:
            score += 5.0
        else:
            rtoks = set(rt.split()[:24])
            ctoks = set(ct.split()[:24])
            if rtoks and ctoks:
                overlap = len(rtoks & ctoks) / max(1, min(len(rtoks), len(ctoks)))
                score += overlap * 4.0
        if score < 4.0:
            continue
        gid = int(ch.get("_global_index"))
        if best is None or score > best[0]:
            best = (score, gid)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus_chunks_parts")
    ap.add_argument("--rich", default="output/rich_chunks/rich_chunks.jsonl")
    ap.add_argument("--index", default="corpus_vector_index")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from clinical_knowledge.vector_index import load_index

    corpus_parts = _jsonl_parts((ROOT / args.corpus).resolve())
    rich_path = (ROOT / args.rich).resolve()
    index_dir = (ROOT / args.index).resolve()
    out_path = Path(args.out).expanduser() if args.out else index_dir / "chunk_id_global.json"

    if not corpus_parts:
        print("Нет corpus JSONL", file=sys.stderr)
        return 1
    if not rich_path.is_file():
        print(f"Нет rich: {rich_path}", file=sys.stderr)
        return 2
    if not (index_dir / "meta.json").is_file():
        print(f"Нет индекса: {index_dir}", file=sys.stderr)
        return 3

    load_index(index_dir)
    corpus_by_path = _load_corpus_by_path(corpus_parts)
    rich_rows = _load_rich_rows(rich_path)

    # Базовая карта corpus chunk_id → global (только indexed)
    from clinical_knowledge import vector_index as vi

    base_map: dict[str, int] = dict(vi._chunk_id_to_global or {})
    mapped = 0
    scanned = 0
    for rich in rich_rows:
        scanned += 1
        cid = str(rich.get("chunk_id") or "")
        path = _norm_path(str(rich.get("source_path") or ""))
        if not cid or not path:
            continue
        if cid in base_map:
            continue
        corpus_rows = corpus_by_path.get(path) or []
        if not corpus_rows:
            tail = path.split("/")[-1]
            for cp, rows in corpus_by_path.items():
                if cp.endswith(tail):
                    corpus_rows = rows
                    break
        gid = _pick_match(rich, corpus_rows)
        if gid is None:
            continue
        # Только если global index реально в vector index
        if vi._global_to_local and gid not in vi._global_to_local:
            continue
        base_map[cid] = int(gid)
        mapped += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base_map, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out_path),
                "corpus_chunk_ids": len(vi._chunk_id_to_global or {}),
                "rich_mapped": mapped,
                "total": len(base_map),
                "rich_scanned": scanned,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
