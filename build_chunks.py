#!/usr/bin/env python3
"""
Строит chunks.json — скользящие фрагменты по клинической части PDF для полнотекстового поиска.
Без этого браузер либо тянет гигантский corpus.json, либо ищет по урезанному тексту.

Запуск после: python3 extract_corpus.py && python3 build_chunks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from build_structured_index import DIAG_PAT, TREAT_PAT, clinical_start

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.json"
OUT = ROOT / "chunks.json"

# Размер фрагмента и перекрытие (символы)
CHUNK_SIZE = 880
OVERLAP = 120
# Ограничения на размер индекса для статического хостинга
MAX_BODY_CHARS = 52000  # с клинического начала — покрывает диагностику+лечение у большинства КП
MAX_CHUNKS_PER_DOC = 36


def kind_for_chunk(text: str) -> str:
    low = text.lower()
    d = len(DIAG_PAT.findall(low))
    t = len(TREAT_PAT.findall(low))
    if d >= t + 2:
        return "diagnostic"
    if t >= d + 2:
        return "treatment"
    return "general"


def sliding_chunks(body: str) -> list[str]:
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    if len(body) <= CHUNK_SIZE:
        return [body]
    step = max(CHUNK_SIZE - OVERLAP, 200)
    out: list[str] = []
    pos = 0
    while pos < len(body) and len(out) < MAX_CHUNKS_PER_DOC:
        piece = body[pos : pos + CHUNK_SIZE].strip()
        if len(piece) >= 80:
            out.append(piece)
        pos += step
    return out


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(f"Нет {CORPUS}. Запустите: python3 extract_corpus.py")

    rows = json.loads(CORPUS.read_text(encoding="utf-8"))
    out: list[dict] = []

    for r in rows:
        path = r["path"]
        text = (r.get("text") or "").strip()
        parts = path.split("/")
        cat = parts[1] if len(parts) > 1 else ""
        title = Path(path).stem.replace("_", " ")

        if not text:
            continue

        start = clinical_start(text)
        clinical = text[start:] if start else text
        body = clinical[:MAX_BODY_CHARS]
        chunks = sliding_chunks(body)

        for i, ch in enumerate(chunks):
            cid = path + "::" + str(i)
            out.append(
                {
                    "id": cid,
                    "path": path,
                    "category": cat,
                    "title": title,
                    "chunk_index": i,
                    "kind": kind_for_chunk(ch),
                    "text": ch,
                }
            )

    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Чанков: {len(out)} из {len(rows)} протоколов → {OUT}")


if __name__ == "__main__":
    main()
