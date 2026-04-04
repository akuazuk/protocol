#!/usr/bin/env python3
"""Строит protocol_meta.json: path → рубрика и человекочитаемая специальность (по папке Минздрава)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "protocols.json"
OUT = ROOT / "protocol_meta.json"

# Соответствие slug каталога → профиль врача / специальность
SPECIALTY_RU: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Кардиология и болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология, нарушения обмена веществ",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Онкология и новообразования",
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
    "zabolevaniya-perinatalnogo-perioda": "Заболевания перинатального периода",
}


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Нет {SRC}. Запустите: python3 build_index.py")
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        path = r.get("path") or ""
        if not path:
            continue
        cat = (r.get("category") or "").strip()
        out[path] = {
            "category": cat,
            "specialty_ru": SPECIALTY_RU.get(cat, cat),
            "title": (r.get("title") or "").strip(),
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(out)} записей → {OUT}")


if __name__ == "__main__":
    main()
