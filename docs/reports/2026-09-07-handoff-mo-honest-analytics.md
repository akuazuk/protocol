# Handoff: honest analytics U07-U14

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-honest-analytics-u07-u14-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-honest-analytics-pc1`
- Base: `07dae5421f3401f5725c8781401f6c66e3da53f5`
- BUILD_VERSION: `2026-09-07-030758Z-honest-analytics`
- PR: [#233](https://github.com/akuazuk/protocol/pull/233)

## Реализация

- U07: семейства лекарств и лаборатории называют автоматический результат сигналом и
  кандидатом на разбор, а не подтверждённым дефектом.
- U08: кольца зон отдельно показывают value и `Оценено: n/N`; при `n=0` вместо
  процента выводится `Не оценено`.
- U09: family KPI и таблицы показывают абсолютные `n/N`; доля группы отделена от
  вклада в период.
- U10: сравнение не использует размер всей группы вместо evaluated N. Пока
  `evaluated_cases` не опубликован, ranking подавлен с явной причиной
  `Оценимый знаменатель недоступен`.
- U11: ненулевые доли меньше 0,1% показываются как `<0,1%` вместе с абсолютным n.
- U12: технический finding code убран из основного столбца и доступен в раскрываемом
  блоке `Технические данные`.
- U13: cases, family dashboards и reports получили локальные error states и не
  очищают ранее загруженные данные; retry ограничен двумя попытками. Старое
  общее сообщение об ошибке загрузки cases удалено.
- U14: используются названия `Проверка назначений` и `Инструкции препаратов`.
- Report cards показывают cohort, methodology и status.

Клинические пороги, веса и primary/shadow флаги не менялись.

## Проверки

- Focused Python suite: 62 passed.
- Полный текущий browser smoke: 15 passed.
- Synthetic browser acceptance: tiny nonzero, technical details, evaluated denominator
  guard, bounded retry и `n=0 → Не оценено` - успешно.
- `node --check frontend/web/shared/mo-app.js` - успешно.
- `python3 scripts/normalize_ui_dashes.py` выполнен; несвязанные изменения отменены.
- `git diff --check` - успешно.
- IDE diagnostics - ошибок нет.
- Локальный `ruff` отсутствует; проверка остаётся обязательной в CI.

## Production baseline

- Production SHA: `07dae5421f3401f5725c8781401f6c66e3da53f5`.
- Production version: `2026-09-06-191123Z-filter-draft-guard`.
- `/health/live` успешен.
- U07-U14 ещё не merged и не deployed.

## Следующий безопасный шаг

После публикации: required CI → merge через GitHub → deploy exact `origin/main` →
production browser smoke для U07-U14 и reports без generic cases error.

## Не трогать параллельно

- `clinical_knowledge/mo_backend.py`
- `frontend/web/methodist/mis-kz-quality.html`
- `frontend/web/shared/mo-app.js`
- `frontend/web/shared/mo-ui.css`
- `rag_server.py`, кроме согласованного разрешения одной строки `BUILD_VERSION`
