#!/usr/bin/env python3
"""
Делит output/chunks/chunks.jsonl на части < лимита GitHub (~100 МБ).
Части: corpus_chunks_parts/chunks.part.NNN.jsonl

  python3 split_chunks_jsonl.py
  cat corpus_chunks_parts/chunks.part.*.jsonl > chunks_merged.jsonl  # сборка
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "chunks" / "chunks.jsonl"
OUT_DIR = ROOT / "corpus_chunks_parts"
# Запас до 100 МБ (байты на UTF-8)
MAX_PART_BYTES = int(os.environ.get("CHUNK_PART_MAX_BYTES", str(45 * 1024 * 1024)))


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Нет файла: {SRC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # очистить старые части
    for p in OUT_DIR.glob("chunks.part.*.jsonl"):
        p.unlink()

    part_idx = 0
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal part_idx, buf, size
        if not buf:
            return
        name = OUT_DIR / f"chunks.part.{part_idx:03d}.jsonl"
        name.write_text("".join(buf), encoding="utf-8")
        print(f"→ {name} ({len(buf)} строк, {size} байт)")
        part_idx += 1
        buf = []
        size = 0

    with SRC.open(encoding="utf-8") as f:
        for line in f:
            b = len(line.encode("utf-8"))
            if buf and size + b > MAX_PART_BYTES:
                flush()
            buf.append(line)
            size += b
    flush()
    print(f"Готово: {part_idx} частей в {OUT_DIR}")


if __name__ == "__main__":
    main()
