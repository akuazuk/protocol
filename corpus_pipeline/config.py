"""Пути и константы пайплайна корпуса."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = Path(os.environ.get("CORPUS_PDF_ROOT", str(ROOT / "minzdrav_protocols")))
OUTPUT_ROOT = Path(os.environ.get("CORPUS_OUTPUT_ROOT", str(ROOT / "output")))
OUT_DOCS = OUTPUT_ROOT / "documents"
OUT_CHUNKS = OUTPUT_ROOT / "chunks"
OUT_TABLES = OUTPUT_ROOT / "tables"
OUT_ENTITIES = OUTPUT_ROOT / "entities"
OUT_REGISTRY = OUTPUT_ROOT / "registry"

# Минимум символов на странице, ниже — считать слой «плохим» и пробовать OCR
MIN_CHARS_PER_PAGE_FOR_NATIVE = 80
# Минимальная уверенность при эвристике (0–1)
DEFAULT_EXTRACTION_CONFIDENCE = 0.75

SPECIALTY_FROM_FOLDER: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Онкология",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Перинатология",
}
