"""Раздел бизнес-плана: непрерывное дообучение моделей Protocol."""
from __future__ import annotations

CONTINUOUS_ML_INTRO = """
Protocol спроектирован как гибридная система: детерминированное ядро (482+ правил, send_gate, CISZ)
принимает решение о подписи КЗ, а дообучаемые модели улучшают поиск протоколов, сопоставление медтерминов
и структурирование данных - без «чёрного ящика» в блокировке ЭЦП.

Отличие от свободных LLM-чатов и зарубежных CDSS: корпус ~478 КП Минздрава РБ + цикл human-in-the-loop
от методслужбы Кравиры + локальное развёртывание моделей в контуре Республики Беларусь (on-prem, без выноса ПДн на L0).

Статус на 2026: в production - правила, semantic RAG, опциональный LLM API; каталог ml/ - каркас MLOps
(сбор feedback, export_training_feedback.py, заглушки train/eval). Fine-tune и деплой локальных моделей - фаза 1+ roadmap.
""".strip()

ML_PRINCIPLES_TABLE = [
    ("send_gate / gate_score", "Только правила", "Не обучается end-to-end"),
    ("CISZ critical gaps", "FHIR BY чек-лист", "Версионирование нормативки"),
    ("RAG retrieval", "Embedder + reranker", "Fine-tune / LoRA ежемесячно"),
    ("required_exam в КЗ", "Entailment matcher", "Метки методиста"),
    ("Summary cards КП", "Structured extractor", "Approved drafts → student"),
    ("L2 пояснения", "Explainer (optional)", "JSON compliance → текст"),
]

ML_ROADMAP_TABLE = [
    ("Фаза 0", "Q2-Q3 2026", "kz_quality_scores, feedback JSONL, export_training_feedback.py", "Пилот Кравира"),
    ("Фаза 1", "Q4 2026", "Local embedder (e5/bge-m3) вместо облачного embed API", "On-prem RAG"),
    ("Фаза 2", "2027 H1", "Cross-encoder reranker + medical entailment", "−ложные L0 срабатывания"),
    ("Фаза 3", "2027 H2", "30-100 approved summary cards, hybrid mode", "Качество vs auto-rules"),
    ("Фаза 4", "2028+", "Ежемесячный LoRA-retrain + A/B eval", "Data moat, масштаб сети ОЗ"),
]

ML_DATA_CYCLE = [
    "Production L0-L2 пишет обезличенные события (hash текста, scores, latency, rule overrides).",
    "Методист исправляет retrieval и rule outcomes → data/ml/feedback/*.jsonl.",
    "export_training_feedback.py формирует ml/datasets/ (retrieval, entailment, kz_regression).",
    "Fine-tune (LoRA) на GPU в контуре РБ; eval vs golden queries и consult_gold.jsonl.",
    "Успешный прогон → ml/registry/model_manifest.json → деплой sidecar (RAG_EMBED_BACKEND=local).",
]

ML_COMPETITION_NOTE = """
Для конкурса Белинфонд: непрерывное дообучение - конкурентное преимущество на национальном рынке
(корпус КП РБ + поток 25 000 КЗ/мес в Кравире как источник меток). Инвестиции сертификата ГКНТ:
интеграция МИС, on-prem L0, запуск ML-контура и масштабирование на 5-10 частных ОЗ.
""".strip()

ML_APPENDIX_TABLE = [
    ("ml/README.md", "Архитектура ML и принципы"),
    ("ml/configs/default.json", "Модели, пороги eval, расписание retrain"),
    ("scripts/export_training_feedback.py", "Экспорт датасетов из пилота"),
    ("data/ml/feedback/", "Сырые события без ПДн"),
    ("eval/golden_queries.prod.jsonl", "Эталон RAG (18 запросов)"),
    ("data/gastro_mvp/consult_gold.jsonl", "Регрессия KZ (9 кейсов)"),
]
