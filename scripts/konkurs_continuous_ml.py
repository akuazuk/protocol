"""Раздел бизнес-плана: непрерывное дообучение моделей Protocol."""
from __future__ import annotations

CONTINUOUS_ML_INTRO = """
Protocol спроектирован как гибридная система: детерминированное ядро (482+ правил, send_gate, CISZ)
принимает решение о подписи КЗ, а дообучаемые модели улучшают поиск протоколов, сопоставление медтерминов
и структурирование данных - без «чёрного ящика» в блокировке ЭЦП.

Отличие от свободных LLM-чатов и зарубежных CDSS: корпус ~478 КП Минздрава РБ + цикл human-in-the-loop
от методслужбы Кравиры + локальное развёртывание моделей в контуре Республики Беларусь (on-prem, без выноса ПДн на L0).

Статус на 2026: в production - правила, semantic RAG, опциональный LLM API. Выполнен пилотный fine-tune
local embedder (e5-small, 313 пары, эксперимент embedder_exp_001) и A/B на consult_gold + golden RAG.
Деплой локальной модели в production retrieve - фаза 1 (Q4 2026).
""".strip()

ML_EXPERIMENT_EMBEDDER_TABLE = [
    ("MRR@10, 5-fold CV (seed 313 пар)", "0,12", "0,41", "+0,29"),
    ("Recall@1, CV", "~8%", "~28%", "+20 п.п."),
    ("Golden hold-out MRR@10 (13 запр.)", "0,17", "0,33", "+0,16"),
    ("Время обучения (CPU)", "-", "~83 мин", "5-fold + final"),
]

ML_AB_KZ_TABLE = [
    ("Rule checker consult_gold (9 КЗ)", "100%", "100%", "0"),
    ("RAG по тексту КЗ, релевантный КП в топ-5", "100%", "100%", "0"),
    ("Golden RAG retrieve() (18 запр.)", "77,8%", "77,8%", "0"),
]

ML_AB_INTERPRETATION = """
Интерпретация A/B: детерминированный разбор КЗ (send_gate, rule_checker) не использует embedder - без изменений.
На гастро-эталоне полный retrieve() уже находит нужные КП и с baseline e5: тексты КЗ богаты диагнозом и МКБ.
Прирост MRR в офлайн-эксперименте проявился на seed-парах title→path; для продакшена следующий шаг -
hard negatives от методиста, cross-encoder reranker и фиксация 4 провалов golden (E11, K85, I50, негатив).
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
    ("Фаза 0", "Q2-Q3 2026", "feedback JSONL, export, embedder_exp_001, A/B consult_gold", "Пилот Кравира"),
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
    ("ml/experiments/embedder_exp_001/", "Fine-tune e5: MRR 0,12→0,41"),
    ("ml/experiments/ab_kz_embedder/", "A/B: RAG и consult_gold"),
    ("scripts/run_embedder_experiment.py", "Пайплайн обучения embedder"),
    ("scripts/run_ab_embedder_kz.py", "A/B baseline vs fine-tune в retrieve()"),
    ("scripts/export_training_feedback.py", "Экспорт датасетов из пилота"),
    ("data/ml/feedback/", "Сырые события без ПДн"),
    ("eval/golden_queries.prod.jsonl", "Эталон RAG (18 запросов)"),
    ("data/gastro_mvp/consult_gold.jsonl", "Регрессия КЗ (9 кейсов)"),
]
