# Handoff: W2 рабочий стол МО - разбор на всю страницу

Дата: 2026-09-20
Ветка: `cursor/mo-workspace-w2-pc1`
Worktree: `/private/tmp/protocol-task-mo-workspace-w2-1`
Base: `origin/main` `b3a37e3b` (W1 в проде)

## Сделано

- Разбор занимает область `.app` справа от сайдбара. Список и шапка фильтров прячутся, не сжимаются в 680 px.
- Порог 1440 split снят. `wideInspector()` всегда false.
- URL: `?open=<visit_id>` на текущем экране (фильтры сохраняются). Back / «К списку» / Escape закрывают разбор.
- Первый экран: 3 оценки, «Что не так», текст МО, решение. История / КП / №55 в «Подробнее».
- Скрытый `#period` убран из a11y (`aria-hidden`, `tabindex=-1`) - хвост W1.
- `BUILD_VERSION` `2026-09-20-095731Z-mo-workspace-w2`.

## Не сделано

- FastAPI `GET /methodist/mo/cases/{id}`: `rag_server.py` занят #186 / #113. Refresh пути `/cases/123` даст 404, пока нет декоратора. Клиент уже читает pathname, если маршрут появится.
- W3-W7.
- План-файл `docs/plans/2026-09-20-mo-analytics-workspace-v1.md` занят #261 - правки ниже, внести в план при merge #261.

## Поправки в план (замечено на проде и при W2)

1. **W2b.** Когда `rag_server.py` свободен: декоратор `@app.get("/methodist/mo/cases/{case_id}")` на `_serve_methodist_mo`, тот же HTML. Тогда `syncUrl` может пушить pathname, не только `?open=`.
2. **W1 follow-up закрыт в W2:** `#period` больше не в a11y-дереве. Facet inner «Применить» оставить - это apply меню, не второй warehouse Apply.
3. **W3.** После полного экрана разбора кольца Обзора всё ещё про день при зерне Месяц - не откладывать.
4. **Не смешивать** W4 (МИС) с этим PR.

## Проверки

`pytest` w1/w2/inspector/ui-phase2/frontend-structure/yesterday/plan-close/workspace-routes - passed.

## Следующее

Merge W2 → GCE `SYNC_PROTOCOL_CORPUS=0` → на 1280 и 1440 открыть случай: текст ≥600 px, Back к тем же датам. Затем W3.
