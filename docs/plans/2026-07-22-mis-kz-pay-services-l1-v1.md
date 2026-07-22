# MIS KZ L1: новые данные июля + срезы pay/services + UI (v1)

**Статус:** active  
**Дата:** 2026-07-22  
**Преемник для:** `2026-07-21-mis-kz-llm-progress-full-report-v1.md` (архивирован)

## Контекст

В БД за 2026-07-21 появились новые КЗ (~488). Выгрузка июля: **8412** строк / **8123** уникальных `visit_id` (было 7916 / 7648). Нужны срезы по `pay_type` и услугам, правило multi-KZ, эвристика exams/treatment, очередь LLM, pastel UI без serif.

## Что изменено в проде / репо

- `clinical_knowledge/mis_pay_type.py` - ярлыки 0/2/3/12
- `scripts/run_mis_protocol_l1_batch.py` - выбор богатого КЗ, `n_kz_per_visit`, pay/services/core/fill, `llm_review_queue`
- `clinical_knowledge/mis_kz_quality.py` - API отдаёт новые поля
- `mis-kz-quality.html` - pastel, Outfit/DM Sans, новые таблицы/бары

## Метрики

| Метрика | Было | Стало (цель) |
|---------|------|--------------|
| Строк CSV июля | 7916 | 8412 |
| Уник. визитов L1 | 7648 | ~8123 |
| Срезы pay/services | нет | да |
| Core overall (без exams/treatment) | нет | в summary + KPI |
| UI шрифты | Fraunces+DM Sans | только sans (Outfit/DM Sans) |
| LLM queue | UI only | `llm_review_queue` + json |

## Шаги

1. [x] Проверка БД за вчера + re-export 8412
2. [x] P1-P5 код: pay/services, labels, multi-KZ, exams heuristic, LLM queue
3. [x] P7 UI pastel sans
4. [x] Upload CSV на Render + L1 `--resume --direct` (475 новых, итого 8123)
5. [x] Rebuild summary с enrich CSV, анализ качества, commit+push
6. [~] P6: regex queue 17 на месте; precompute top-20 section overviews запущен на Render (`/var/data/ml/section_overviews`)

## Риски

- PHI в cases.jsonl / patient_id в worst_visits - не коммитить jsonl
- L1 на Render: только resume, не полный пересчёт без нужды
- `normalize_ui_dashes.py` без аргументов трогает весь репо - не запускать так
