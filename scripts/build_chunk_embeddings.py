#!/usr/bin/env python3
"""Офлайн-эмбеддинги для чанков корпуса (Gemini retrieval_document).

Пример:
  python3 scripts/build_chunk_embeddings.py --input corpus_chunks_parts --output corpus_chunks_parts
  python3 scripts/build_chunk_embeddings.py --dry-run --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_load import load_project_env

load_project_env(ROOT)


def _chunk_files(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    parts = sorted(inp.glob("chunks.part.*.jsonl"))
    if parts:
        return parts
    return sorted(inp.glob("*.jsonl"))


def _embed_one(model: str, text: str) -> list[float]:
    import rag_server as rs

    return rs._gemini_embed_one(model, text[:8000], "retrieval_document")


def _text_for_embed(row: dict) -> str:
    t = (row.get("embedding_ready_text") or row.get("text") or "").strip()
    return t[:8000]


def process_file(
    src: Path,
    dst: Path,
    *,
    model: str,
    resume_ids: set[str],
    limit: int,
    dry_run: bool,
) -> dict[str, int]:
    stats = {"read": 0, "skipped": 0, "embedded": 0, "errors": 0}
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            stats["errors"] += 1
            continue
        if not isinstance(row, dict):
            continue
        stats["read"] += 1
        cid = str(row.get("chunk_id") or "")
        if row.get("embedding") and isinstance(row["embedding"], list):
            stats["skipped"] += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))
            continue
        if cid and cid in resume_ids:
            stats["skipped"] += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))
            continue
        if limit > 0 and stats["embedded"] >= limit:
            lines_out.append(json.dumps(row, ensure_ascii=False))
            continue
        text = _text_for_embed(row)
        if len(text) < 20:
            stats["skipped"] += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))
            continue
        if dry_run:
            stats["embedded"] += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))
            continue
        try:
            vec = _embed_one(model, text)
            row["embedding"] = vec
            row["embedding_model"] = model
            row["embedding_dim"] = len(vec)
            stats["embedded"] += 1
            if cid:
                resume_ids.add(cid)
        except Exception:
            stats["errors"] += 1
        lines_out.append(json.dumps(row, ensure_ascii=False))
        if stats["embedded"] % 50 == 0 and stats["embedded"]:
            time.sleep(0.05)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        tmp.replace(dst)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Офлайн embeddings для JSONL-чанков")
    parser.add_argument("--input", default="corpus_chunks_parts", help="Файл или каталог JSONL")
    parser.add_argument("--output", default="", help="Каталог вывода (по умолчанию = input)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"))
    parser.add_argument("--state", default="output/embed_build_state.json")
    parser.add_argument("--limit", type=int, default=0, help="Макс. новых embed за запуск (0 = все)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inp = (ROOT / args.input).resolve()
    out_root = (ROOT / (args.output or args.input)).resolve()
    state_path = (ROOT / args.state).resolve()
    resume_ids: set[str] = set()
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            resume_ids = set(st.get("done_chunk_ids") or [])
        except Exception:
            pass

    files = _chunk_files(inp)
    if not files:
        print(f"Нет JSONL в {inp}", file=sys.stderr)
        return 1

    total = {"read": 0, "skipped": 0, "embedded": 0, "errors": 0}
    for src in files:
        rel = src.name
        dst = out_root / rel if out_root.is_dir() else out_root
        st = process_file(
            src,
            dst,
            model=args.model.strip(),
            resume_ids=resume_ids,
            limit=max(0, args.limit - total["embedded"]) if args.limit else 0,
            dry_run=args.dry_run,
        )
        for k in total:
            total[k] += st[k]
        print(f"{rel}: {st}")

    if not args.dry_run:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "done_chunk_ids": sorted(resume_ids),
                    "stats": total,
                    "model": args.model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print("Итого:", total)
    return 0 if total["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
