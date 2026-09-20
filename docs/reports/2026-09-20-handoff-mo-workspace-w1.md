# Handoff: W1 рабочий стол МО - одна выборка

Дата: 2026-09-20
Ветка: `cursor/mo-workspace-w1-pc1`
Worktree: `/private/tmp/protocol-task-mo-workspace-w1-1`
Base: `origin/main` `f0a72b2c`

## Сделано

- Календарь и пресеты периода в шапке (`#period-strip`), без «Произвольный» в details.
- `date_from`/`date_to` всегда уходят в API; бэкенд берёт явные даты даже при `period=month`.
- Убраны `#filters-apply` / `#filters-cancel`. Сброс: «Сбросить всё».
- Col-filters сняты с таблиц склада (`serverSort`).
- Placeholder поиска: «Врач, МКБ, visit_id». Empty: «Код не найден».
- `BUILD_VERSION` `2026-09-20-085541Z-mo-workspace-w1`.

## Не сделано

- Разбор на всю страницу (W2).
- Поиск МИС (W4).
- План PR #261 ещё не в main (не хватает required checks).

## Проверки

`pytest` W1 + cohort/metrics/frontend/ui-phase2/visit-search/backend/dimensions - passed.

## Следующее

После merge и GCE smoke: календарь без «Фильтры», день режет total. Затем W2.
