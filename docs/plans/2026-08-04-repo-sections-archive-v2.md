# Разделы продукта и безопасная архивация репозитория (v2)

Дата: 2026-08-04  
Статус: active  
Предшественник: `2026-07-29-repo-structure-cleanup-v1.md`  
Связанные: `2026-08-03-ci-release-concurrency-v3.md` (completed),  
`2026-08-04-mo-runtime-stabilization-v1.md` (active на task-ветке)

---

## Контекст

Локальный checkout ~7.9 GB / ~63k файлов. Продукт - монорепо вокруг одного
FastAPI (`rag_server.py` / `backend.server:app`) с несколькими разделами, но в корне
смешаны UI leftovers, corpus-артефакты, конкурсный пакет, ML batch dumps и плоский
`scripts/`. Cleanup v1 уже создал `frontend/web`, `backend/`, `scripts/{ops,data,dev,deploy}`,
но не довёл архивацию и карту разделов до конца.

## Цель

1. Зафиксировать карту разделов продукта.
2. Убрать из рабочей навигации то, что не нужно для daily development / prod.
3. Не ломать prod URL, Render cold start и CI.
4. Дальше (фазы 2+) группировать код и docs по разделам малыми шагами.

## Карта разделов продукта

| Раздел | Смысл | Канон сейчас |
|--------|-------|--------------|
| Doctor search | Поиск протоколов, МКБ | `frontend/web/doctor`, `rag_server.py`, `clinical_knowledge/protocol_*` |
| Consult review | Проверка КЗ L0/L1/L2 | `consult_review_pipeline.py`, doctor UI, `clinical_knowledge/consult_*` |
| Methodist / МО | Ежедневный ETL + BI | `mo_*.py`, `frontend/web/methodist`, `scripts/run_mo_*` |
| Patient B2C | Отчёт пациенту | `frontend/web/patient`, `patient_*.py`, `patient-app/` |
| Corpus pipeline | PDF МЗ → chunks → индексы | `corpus_pipeline/`, `minzdrav_protocols/`, `output/` |
| MIS / КЗ data | Выгрузка MariaDB, L1 | `data/mis_protocol`, `export_mis_*`, `epam/` |
| ML | Обучение и эксперименты | `ml/`, `data/ml/` |
| Ops / deploy | Render, git guards | `scripts/ops/`, `.github/`, `docs/deploy/` |
| Docs | Планы, архитектура, отчёты | `docs/plans`, `docs/deploy`, `docs/reports` |

## Что изменено в production

До merge этой задачи production остаётся на текущем `origin/main`.
Фаза 1 не меняет runtime-поведение API/UI; только структуру репозитория и docs.

## Метрики

| Метрика | Было | Цель фазы 1 | Цель фазы 2+ |
|---------|------|-------------|--------------|
| `docs/konkurs/` в рабочей docs-навигации | да | в `archive/docs/konkurs` | archive |
| Исторические `ml/experiments/batch_*` в рабочей зоне | да | в `archive/ml-experiments` | archive |
| Пустые stub-каталоги (`export/`, `checkpoints/`) | да | 0 или README-указатель | 0 |
| Официальный план разделов | нет | 1 файл v2 | active |
| Root CSS/JS leftovers | да | без изменений в фазе 1 | в `frontend/web/shared` |
| Плоский `scripts/*.py` (~180) | да | без переноса в фазе 1 | домены scripts |

## Фазы

### Фаза 1 - безопасная архивация (эта итерация)

- [x] Оформить этот план и индекс `docs/plans/README.md`.
- [x] `docs/konkurs/` → `archive/docs/konkurs/` (+ `archive/README.md`, stub `docs/konkurs/README.md`).
- [x] Исторические `ml/experiments/*` → `archive/ml-experiments/` (оставить `ml/experiments/README.md` для новых прогонов).
- [x] Пустые `export/`, `checkpoints/` отсутствовали в дереве / удалены.
- [x] Усилить `.gitignore`: `.local-archive/`, cache/logs, которые не должны попадать в git.
- [x] Обновить ссылки в docs/scripts, которые указывали на старые пути.
- [x] Smoke: `ruff` на затронутое, `scripts/ops/smoke_repo_layout.sh`, точечные тесты.
- [x] `BUILD_VERSION`, commit, push, PR.

Не делать в фазе 1:

- перенос root CSS/JS/logos;
- перенос root `build_*.py`;
- split `rag_server.py`;
- удаление corpus JSON из корня;
- физический перенос GB-локальных `output/` / `corpus_vector_index/` (только ignore/документация).

### Фаза 2 - корень по разделам

- Root CSS/JS/logos → `frontend/web/shared/`.
- Root legacy build/verify → только `scripts/data|dev/py` (+ shim на 1 релиз).
- Группировка `scripts/mo`, `scripts/mis`, `scripts/corpus`.
- `docs/{architecture,product,deploy,plans,reports}`.

### Фаза 3 - data plane

```text
data/committed/   # catalog, summaries, benchmarks
data/runtime/     # medical_exams - never git
data/cache/       # chunk_qa, build state - gitignore
```

Отдельное решение: root corpus JSON остаются в git или переезжают на disk-only.

### Фаза 4 - backend split

Маршруты по разделам из `rag_server.py` без big-bang. `clinical_knowledge/` пока целый.

### Фаза 5 - гигиена навсегда

- hygiene/CI запрет новых тяжёлых файлов в корне;
- карта разделов в `AGENTS.md` / README.

## Целевой top-level (ориентир, не big-bang)

```text
Protocol/
  apps or frontend/web/{doctor,methodist,patient,shared}
  backend/
  clinical_knowledge/
  pipelines/          # будущее: mo_daily, corpus, mis_export
  scripts/{ops,deploy,data,dev,mo,mis,corpus}
  config/ data/ tests/ eval/ docs/
  archive/{docs/konkurs,ml-experiments,legacy-root-scripts}
  .local-archive/     # только локально, gitignore
```

## Риски

| Риск | Митигация |
|------|-----------|
| Сломаются ссылки на konkurs / REPORT.md | Обновить docs; в `ml/experiments/README` указать archive |
| Скрипты пишут в старый `ml/experiments/batch_*` | Оставить каталог для новых прогонов; в archive только история |
| Случайно затронуть prod URL | Фаза 1 без переноса frontend/backend runtime |
| Конфликт с MO Phase A веткой | Не трогать `mo_daily.py`, `publish_mo_to_render.py`, `mo-app.js` |

## Definition of Done фазы 1

1. План v2 в индексе, cleanup v1 помечен archived с преемником.
2. `archive/` содержит konkurs и исторические ML experiments.
3. Новые batch-прогоны по-прежнему могут писать в `ml/experiments/`.
4. CI/smoke зелёные; PR открыт от task-ветки.
