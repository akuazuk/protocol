# Handoff: план рабочего стола МО Аналитики

Дата: 2026-09-20
Репозиторий: ветка `cursor/mo-workspace-plan-pc1`, worktree `/private/tmp/protocol-task-mo-workspace-plan-1`
PR: https://github.com/akuazuk/protocol/pull/261
Прод не менялся этой сессией: GCE `2026-09-19-192135Z-mo-table-polish` / `58c956d4`

## Сделано

- Повторный аудит экранов и разбора на 1440.
- API: month=8062, custom день живой, date_from без custom и diagnosis_code игнор, q=гипертон=0, queue_band=13 vs плитка 0.
- Как устроен поиск: только `fact_mo_case` / findings; МИС и `mis_tests` в UI не ищутся; ingest визита - CLI на GCE.
- План: `docs/plans/2026-09-20-mo-analytics-workspace-v1.md` - добавлена рубрика «Поиск МИС» (волна 4), визуал таймлайна анализов, кнопка «Проанализировать», бейджи «В аналитике / Не разобрано». Старые W4-W6 сдвинуты в W5-W7.
- Канвас: `mo-workspace-audit.canvas.tsx` (F8 + pill W4).

## Не сделано

- Код UI/API не менялся, деплоя нет.
- W1 календаря ещё нет.
- Каталог `fact_mis_catalog` и POST ingest из кабинета - только в плане.

## Нельзя параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`, `rag_server.py` Query `/cases`, хром таблиц `ensureTableChrome`. W4 позже затронет новые API mis-search/ingest и пункт меню.

## Следующая команда

```bash
scripts/ops/git_task_start.sh mo-workspace-w1 --pc=1 \
  --branch=cursor/mo-workspace-w1-pc1
```

Только W1. Не начинать полный разбор и не начинать Поиск МИС до merge W1.
