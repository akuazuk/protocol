# Handoff: filter draft guard hotfix

Дата: 2026-09-06

## Состояние

- Branch: `hotfix/mo-filter-draft-guard-agent1-pc1`
- Worktree: `/private/tmp/protocol-hotfix-mo-filter-draft-pc1`
- Base: `b99f3d1c7f2e75496d4491e3e3d2c954ac5bfacb`
- BUILD_VERSION: `2026-09-06-191123Z-filter-draft-guard`
- PR: будет указан после публикации

## Причина и исправление

Production smoke после выпуска #231 показал, что изменение периода внутри открытой
панели могло обновить URL до нажатия `Применить`, если событие `toggle` не успело
создать draft snapshot. `setFilterDraftValue()` теперь дополнительно проверяет открытое
состояние `#filters-panel` и синхронно создаёт snapshot перед записью значения.

## Проверки

- Production воспроизведение на `b99f3d1c`: дефект подтверждён.
- Локальный browser smoke: URL не меняется до Apply, Cancel сохраняет исходный URL.
- Focused pytest: 30 passed.
- `node --check frontend/web/shared/mo-app.js` - успешно.
- `git diff --check` - успешно.

## Production

- Сейчас в production: `b99f3d1c7f2e75496d4491e3e3d2c954ac5bfacb`,
  `2026-09-06-184846Z-mo-ui-shell`.
- Hotfix ещё не merged и не deployed.

Следующий шаг: CI → merge PR → deploy exact `origin/main` → повторить browser smoke.
