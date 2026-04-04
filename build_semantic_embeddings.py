#!/usr/bin/env python3
"""
Строит семантические векторы для протоколов (мультиязычная e5).
Требует: pip install sentence-transformers torch

Выход: embeddings.json — используется в index.html вместе с corpus.json
(запрос в браузере кодируется той же моделью через @xenova/transformers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.json"
OUT = ROOT / "embeddings.json"
MODEL_NAME = "intfloat/multilingual-e5-small"


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(f"Сначала запустите: python3 extract_corpus.py → {CORPUS}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Установите: pip install sentence-transformers torch", file=sys.stderr)
        raise SystemExit(1)

    rows = json.loads(CORPUS.read_text(encoding="utf-8"))
    texts = []
    for r in rows:
        t = (r.get("text") or "").strip()
        title = Path(r["path"]).stem.replace("_", " ")
        # e5: префикс passage для документов
        payload = f"passage: {title}. {t}" if t else f"passage: {title}"
        texts.append(payload)

    print(f"Загрузка модели {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print("Кодирование…")
    emb = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    dim = int(emb.shape[1])
    items = []
    for i, r in enumerate(rows):
        items.append({"path": r["path"], "v": emb[i].tolist()})

    payload = {"dim": dim, "model": MODEL_NAME, "items": items}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Готово: {OUT} ({len(items)} векторов, dim={dim})")


if __name__ == "__main__":
    main()
