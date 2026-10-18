# Handoff: программа точности МО и Composio

Дата: 2026-10-18
Repo: `akuazuk/protocol`
Ветка: `cursor/mo-wave-a-list-truth-agent1-pc1`
Worktree: `/private/tmp/protocol-release-25bb85a3`
База: `6f8da0f8` (`origin/main` на момент ветки)
PR: волна A, деплой после merge.
Прод на старте программы: `2026-10-17-190237Z-mo-fio-search`, git `6f8da0f8`.

## Сделано

Написана программа и шесть документов шагов в `docs/plans/`:

- `2026-10-18-mo-accuracy-ux-program-v1.md`
- `2026-10-18-mo-list-truth-v1.md`
- `2026-10-18-mo-case-review-clarity-v1.md`
- `2026-10-18-mo-grade-signals-v1.md`
- `2026-10-18-mo-screen-layout-v1.md`
- `2026-10-18-mo-filter-contract-v1.md`
- `2026-10-18-mo-composio-audit-loop-v1.md`
- `2026-10-18-mo-control-inventory-v1.md`
- `2026-10-18-mo-forms-v1.md`
- `2026-10-18-mo-layout-color-v1.md`

Строка индекса добавлена в `docs/plans/README.md`.

Composio: поиск инструментов прошёл, сессия `work`. Firecrawl scrape прода не выполнился, кошелёк Instant пуст. Токен методиста и тексты осмотров в Composio не передавались.

## Не сделано

Волна A в коде этой ветки: колонка «Причина» и первая фраза «Почему так» берут заголовок живого замечания. Оценка и склад не меняются. Волны B-D ещё не в коде. Правило «Критично» не менялось. Склад не пересчитывался. Фикстура вёрстки не собрана. На прод волна A ещё не выехала, пока PR не влит и `/api/version` не совпал.

Второй проход 2026-10-18 дополнил перечень контролов. Верх склада на сверке `2026-10-17`. Месяц по умолчанию: 6993 клинических, важно 95, критично 0. Команды ⌘K для вчера / 7 дней / месяца даты не переставляют.

## Следующий безопасный шаг

Волна A отдельной веткой от свежего `origin/main`: причина в списке и первая фраза «Почему так» равны заголовку замечания. Оценку не менять.

Волна E не стартует без отдельного согласия поднять Major не топический и high-alert без дозы в «Критично», вместе с примерно 15 Major вне текущих «Важно».

Волна G не стартует, пока нет кредитов Instant или своего аккаунта Firecrawl / Browser Use.

## Не трогать параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`, `clinical_knowledge/mo_zone_scores.py`, `clinical_knowledge/mo_overall_grade.py`, `clinical_knowledge/mo_backend.py`, `docs/plans/README.md`.
