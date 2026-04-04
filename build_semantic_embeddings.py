#!/usr/bin/env python3
"""
Строит семантические векторы для протоколов (мультиязычная e5).
Для каждого PDF: усреднение векторов по чанкам (chunks.json) — лучше, чем один усечённый фрагмент.

Требует: pip install sentence-transformers torch

Выход: embeddings.json — в браузере запрос кодируется той же моделью (@xenova/transformers).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRUCTURED = ROOT / "structured_index.json"
CHUNKS = ROOT / "chunks.json"
OUT = ROOT / "embeddings.json"
MODEL_NAME = "intfloat/multilingual-e5-small"

# Согласовано с index.html CATEGORY_LABELS
LABELS: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология и обмен веществ",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Новообразования",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология и фтизиатрия",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация органов и тканей",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Перинатальный период",
}


def payload_from_structured(r: dict, label: str) -> str:
    title = (r.get("title") or "").strip()
    diag = (r.get("diagnosis") or "").strip()
    treat = (r.get("treatment") or "").strip()
    summ = (r.get("summary") or "").strip()
    if len(summ) > 3500:
        summ = summ[:3500] + "…"
    return (
        f"passage: Медицинское направление: {label}. "
        f"Клинический протокол РБ. Название: {title}. "
        f"Диагностика и критерии: {diag}. "
        f"Лечение и тактика ведения: {treat}. "
        f"Дополнительный контекст: {summ}"
    )[:16000]


def main() -> None:
    if not STRUCTURED.is_file():
        raise SystemExit(f"Нет {STRUCTURED}. Запустите: python3 build_structured_index.py")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Установите: pip install sentence-transformers torch", file=sys.stderr)
        raise SystemExit(1)

    rows = json.loads(STRUCTURED.read_text(encoding="utf-8"))

    chunks_by_path: dict[str, list[dict]] = defaultdict(list)
    if CHUNKS.is_file():
        for c in json.loads(CHUNKS.read_text(encoding="utf-8")):
            chunks_by_path[c["path"]].append(c)
        for path in chunks_by_path:
            chunks_by_path[path].sort(key=lambda x: x.get("chunk_index", 0))

    print(f"Загрузка модели {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)

    texts_for_batch: list[str] = []
    meta: list[tuple[int, int, int]] = []
    # meta: (row_index, chunk_start_in_batch, chunk_count)

    for i, r in enumerate(rows):
        path = r["path"]
        cat = r.get("category") or (path.split("/")[1] if "/" in path else "")
        label = LABELS.get(cat, cat)
        title = (r.get("title") or "").strip()

        chlist = chunks_by_path.get(path, [])
        if chlist:
            start = len(texts_for_batch)
            for c in chlist:
                body = (c.get("text") or "").strip()
                kind = (c.get("kind") or "general").strip()
                texts_for_batch.append(
                    f"passage: Направление: {label}. Протокол: {title}. "
                    f"Фрагмент ({kind}): {body}"
                )
            meta.append((i, start, len(chlist)))
        else:
            texts_for_batch.append(payload_from_structured(r, label))
            meta.append((i, len(texts_for_batch) - 1, 1))

    print("Кодирование…")
    emb = model.encode(
        texts_for_batch,
        batch_size=8,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    items = []
    for row_i, start, count in meta:
        r = rows[row_i]
        path = r["path"]
        slice_e = emb[start : start + count]
        v = np.mean(slice_e, axis=0)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        items.append({"path": path, "v": v.tolist()})

    payload_out = {
        "dim": int(emb.shape[1]),
        "model": MODEL_NAME,
        "passage_template": "mean_chunk_embeddings",
        "items": items,
    }
    OUT.write_text(
        json.dumps(payload_out, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Готово: {OUT} ({len(items)} векторов, dim={payload_out['dim']})")


if __name__ == "__main__":
    main()
