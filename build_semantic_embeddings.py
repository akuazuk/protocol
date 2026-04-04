#!/usr/bin/env python3
"""
Строит семантические векторы для протоколов (мультиязычная e5).
Текст для эмбеддинга включает направление (рубрику) — лучше сопоставление с запросами по симптомам.

Требует: pip install sentence-transformers torch

Выход: embeddings.json — в браузере запрос кодируется той же моделью (@xenova/transformers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.json"
STRUCTURED = ROOT / "structured_index.json"
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


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(f"Сначала запустите: python3 extract_corpus.py → {CORPUS}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Установите: pip install sentence-transformers torch", file=sys.stderr)
        raise SystemExit(1)

    use_structured = STRUCTURED.is_file()
    if use_structured:
        rows = json.loads(STRUCTURED.read_text(encoding="utf-8"))
    else:
        rows = json.loads(CORPUS.read_text(encoding="utf-8"))

    texts: list[str] = []
    for r in rows:
        path = r["path"]
        cat = r.get("category") if use_structured else (path.split("/")[1] if "/" in path else "")
        label = LABELS.get(cat, cat)

        if use_structured:
            title = (r.get("title") or "").strip()
            diag = (r.get("diagnosis") or "").strip()
            treat = (r.get("treatment") or "").strip()
            summ = (r.get("summary") or "").strip()
            if len(summ) > 3500:
                summ = summ[:3500] + "…"
            payload = (
                f"passage: Медицинское направление: {label}. "
                f"Клинический протокол РБ. Название: {title}. "
                f"Диагностика и критерии: {diag}. "
                f"Лечение и тактика ведения: {treat}. "
                f"Дополнительный контекст: {summ}"
            )
        else:
            fname = Path(path).name
            title = fname.rsplit(".", 1)[0].replace("_", " ")
            body = (r.get("text") or "").strip()
            if len(body) > 6000:
                body = body[:6000] + "…"
            if body:
                payload = (
                    f"passage: Медицинское направление: {label}. "
                    f"Клинический протокол Республики Беларусь. "
                    f"Название: {title}. "
                    f"Фрагмент документа: {body}"
                )
            else:
                payload = (
                    f"passage: Медицинское направление: {label}. "
                    f"Клинический протокол Республики Беларусь. Название: {title}."
                )
        texts.append(payload[:16000])

    print(f"Загрузка модели {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print("Кодирование…")
    emb = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    dim = int(emb.shape[1])
    items = []
    for i, r in enumerate(rows):
        items.append({"path": r["path"], "v": emb[i].tolist()})

    payload_out = {
        "dim": dim,
        "model": MODEL_NAME,
        "passage_template": "structured" if use_structured else "direction+title+body",
        "items": items,
    }
    OUT.write_text(
        json.dumps(payload_out, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Готово: {OUT} ({len(items)} векторов, dim={dim})")


if __name__ == "__main__":
    main()
