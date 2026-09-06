# Handoff: UI shell U01-U06

Дата: 2026-09-06

## Состояние

- Repo: `akuazuk/protocol`
- Branch: `cursor/mo-ui-shell-u01-u06-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-ui-shell-pc1`
- Base: `81d3bf2b8e5d7e66863d34ef5575b6a2b7272140`
- PR: #231
- BUILD_VERSION: `2026-09-06-184846Z-mo-ui-shell`

## Реализовано

- U01: в узкой боковой панели остаются видимыми полные русские названия разделов; иконки исключены из accessibility tree, кнопки получили `aria-label`.
- U02: панели фильтров ограничены высотой viewport, прокручиваются внутри и при необходимости открываются вверх; на мобильном закреплены над нижней навигацией.
- U03: пустой маршрут анализа скрыт до первого drill-down, верхняя панель стала компактнее.
- U04: строка чипов всегда показывает период, сравнение, обязательный clinical cohort и локализованные значения зон, оценки, КП и истории.
- U05: фильтры внутри общей панели редактируются как черновик. API, URL и таблицы меняются только после `Применить`; `Отмена` и закрытие панели отбрасывают черновик. Поиск остаётся отдельным явным submit.
- U06: очередь и список случаев по умолчанию показывают восемь ключевых колонок; общий менеджер колонок доступен с обоих экранов и сохраняет выбор в `localStorage`.
- Пользовательские названия разведены: `Проверка назначений` и `Инструкции препаратов`.

Маршруты, API root, RBAC, deep links и drawer contract не менялись.

## Проверки

- `node --check frontend/web/shared/mo-app.js` - успешно.
- `pytest tests/test_mo_frontend_structure.py tests/test_mo_ui_phase2.py tests/test_mo_dashboard_nav_cleanup.py tests/test_workspace_routes.py -q` - 32 passed.
- `playwright test tests/e2e/mo-smoke.spec.ts` - 15 passed.
- `python3 scripts/normalize_ui_dashes.py` выполнен; несвязанные изменения нормализатора отменены.
- `git diff --check` - успешно.
- IDE diagnostics изменённых UI-файлов - ошибок нет.

## Production baseline

- До этой ветки PR #230 выпущен на GCE.
- Production SHA: `81d3bf2b8e5d7e66863d34ef5575b6a2b7272140`.
- Production version: `2026-09-06-174928Z-protocol-zone-sync`.
- `/health/live`, cases API, reports API и protocol-suggest → matched zone smoke успешны.
- UI shell U01-U06 ещё не merged и не deployed.

## Следующий безопасный шаг

```bash
gh pr checks <PR> --repo akuazuk/protocol --watch
```

После зелёного CI: merge через GitHub, затем GCE deploy только exact `origin/main` и browser smoke для 320, 720, 1024 и 1440 px.

## Не трогать параллельно

- `frontend/web/methodist/mis-kz-quality.html`
- `frontend/web/shared/mo-ui.css`
- `frontend/web/shared/mo-app.js`
- `rag_server.py` до merge этой ветки, кроме согласованного разрешения одной строки `BUILD_VERSION`
