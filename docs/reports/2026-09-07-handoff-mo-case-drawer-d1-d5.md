# Handoff: CASE Review drawer D1-D5

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-case-drawer-d1-d5-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-case-drawer-pc1`
- Base: `b5433e538f9c5be65893bdd09e4589fa3fdef0fb`
- BUILD_VERSION: `2026-09-07-041400Z-case-drawer-d1-d5`
- PR: будет указан после публикации

## Реализация

- D1: header drawer показывает позицию `N из M` в текущем отфильтрованном наборе;
  Previous/Next сохраняют текущий список и page offset.
- D2: desktop остаётся двухколоночным; на ширине до 960 px используются доступные
  вкладки `Документ / Проверка` без горизонтального overflow.
- D3: правая колонка перестроена в порядок: оценка и применимость → замечания →
  история и анализы → критерии. Дополнительные автоматические оценки убраны в details.
- D4: замечания представлены evidence cards: один question id - одна карточка,
  фрагмент МО, требование, статус, переход к полю документа и решение методиста.
  Shadow cards не записывают решение как подтверждённое.
- D5: detail и protocol-suggest используют отдельный case-bound epoch и
  AbortController. Старый ответ A не может перерисовать уже открытый случай B.
- После protocol-suggest строка применимости синхронизируется только по явному
  `matched/applicable`, а не по наличию кандидатов.

Клинические пороги, веса и primary/shadow флаги не менялись.

## Проверки

- Focused Python suite: 48 passed.
- Полный текущий browser smoke: 15 passed.
- Synthetic drawer acceptance: case race, `N из M`, mobile tabs, evidence dedup,
  protocol applicability sync, no horizontal overflow - успешно.
- `node --check frontend/web/shared/mo-app.js` - успешно.
- `python3 scripts/normalize_ui_dashes.py` выполнен; несвязанные изменения отменены.
- `git diff --check` - успешно.
- IDE diagnostics - ошибок нет.

## Production baseline

- Production SHA: `b5433e538f9c5be65893bdd09e4589fa3fdef0fb`.
- Production version: `2026-09-07-030758Z-honest-analytics`.
- `/health/live`, honest analytics API и browser smoke успешны.
- D1-D5 ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge через GitHub → exact-main GCE deploy → production drawer smoke.
После этого отдельным PR выполнить D6-D8: unsaved guard, revision/idempotency save и
финальную accessibility acceptance.

## Не трогать параллельно

- `frontend/web/methodist/mis-kz-quality.html`
- `frontend/web/shared/mo-app.js`
- `frontend/web/shared/mo-ui.css`
- `rag_server.py`, кроме согласованного разрешения одной строки `BUILD_VERSION`
