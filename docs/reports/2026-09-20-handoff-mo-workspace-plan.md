# Handoff: план рабочего стола МО Аналитики

Дата: 2026-09-20
Репозиторий: ветка `cursor/mo-workspace-plan-pc1`, worktree `/private/tmp/protocol-task-mo-workspace-plan-1`
Прод не менялся этой сессией: GCE `2026-09-19-192135Z-mo-table-polish` / `58c956d4`

## Сделано

- Повторный аудит всех основных экранов и разбора на 1440.
- API: month=8062, custom день живой, date_from без custom и diagnosis_code игнор, q=гипертон=0, queue_band=13 vs плитка 0.
- План: `docs/plans/2026-09-20-mo-analytics-workspace-v1.md`.
- Индекс: строка в `docs/plans/README.md`, v1 19.09 -> archived.
- Канвас рядом с чатом: `mo-workspace-audit.canvas.tsx`.

## Не сделано

- Код UI не менялся, деплоя нет.
- W1 календаря ещё нет.

## Нельзя параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`, `rag_server.py` Query `/cases`, хром таблиц `ensureTableChrome`.

## Следующая команда

```bash
scripts/ops/git_task_start.sh mo-workspace-w1 --pc=1 \
  --branch=cursor/mo-workspace-w1-pc1
```

Только W1. Не начинать полный разбор до merge W1.
