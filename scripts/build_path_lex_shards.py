#!/usr/bin/env python3
"""Offline: inverted lex index per rubric (token → chunk_id)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.corpus_path_manifest import _norm_path, _rubric_from_path

_ICD_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", re.IGNORECASE)


def _tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


def _collect_parts(corpus_dir: Path) -> list[Path]:
    parts = sorted(corpus_dir.glob("chunks.part.*.jsonl"))
    if not parts:
        parts = sorted(corpus_dir.glob("*.jsonl"))
    return parts


def build_shards(corpus_dir: Path) -> dict[str, dict[str, set[str]]]:
    """rubric → token → set(chunk_id)."""
    shards: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for part in _collect_parts(corpus_dir):
        with part.open(encoding="utf-8") as f:
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
                path = _norm_path(str(row.get("source_path") or ""))
                cid = str(row.get("chunk_id") or "").strip()
                if not path or not cid:
                    continue
                rub = _rubric_from_path(path)
                if not rub:
                    continue
                text = (row.get("embedding_ready_text") or row.get("text") or "").strip()
                title = str(row.get("protocol_title") or "")
                blob = f"{text} {title}"
                for t in _tokenize_ru(blob):
                    if len(t) >= 2:
                        shards[rub][t].add(cid)
                for m in _ICD_RE.findall(blob):
                    shards[rub][m.upper()].add(cid)
    return shards


def write_shards(shards: dict[str, dict[str, set[str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rub, index in sorted(shards.items()):
        out = {
            "rubric": rub,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tokens": len(index),
            "index": {k: sorted(v) for k, v in index.items()},
        }
        (output_dir / f"{rub}.json").write_text(
            json.dumps(out, ensure_ascii=False),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus_chunks_parts")
    parser.add_argument("--output", default="data/catalog/lex_shards")
    args = parser.parse_args()
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = (ROOT / corpus).resolve()
    if not corpus.is_dir():
        print(f"Нет каталога {corpus}", file=sys.stderr)
        return 1
    shards = build_shards(corpus)
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    write_shards(shards, out)
    print(json.dumps({"rubrics": len(shards), "output": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
