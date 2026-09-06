# Handoff: честное отображение групповых family KPI

Дата: 2026-09-06. Roadmap: A10/A11, frontend.

Branch `cursor/mo-family-group-ui-agent1-pc1`.
Worktree `/private/tmp/protocol-task-mo-family-group-ui-pc1` (locked).
Base при старте: `ff4a811ccea634458fcb1bace14fd58b33a9a657`.

## Требование и реализация

Таблицы врачей и специальностей больше не подписывают вклад в общий период как
долю проблем внутри группы. UI отдельно показывает:

- число МО с замечаниями;
- все МО группы;
- число оценимых случаев или `-`, если coverage не рассчитан;
- долю замечаний внутри группы;
- вклад группы во все МО периода;
- статус допуска к сравнению.

Группы без допуска сортируются по названию, а не по числу проблем. Для n < 20
показывается small-n guard; при неизвестном evaluated denominator явно написано
«без ранга». Старый payload без новых полей не получает выдуманные знаменатели.

## Проверки

- `node --check frontend/web/shared/mo-app.js` - passed.
- `git diff --check` и IDE lint - passed.
- `tests/e2e/mo-smoke.spec.ts` - 15 passed в Chromium.
- Одноразовый browser acceptance с synthetic group payload - 1 passed:
  проверены заголовки, separate rates, small-n/evaluability-unknown labels и
  отсутствие горизонтального overflow страницы при viewport 390x844.

Постоянный browser regression должен публиковаться отдельным test-only PR
уровня 4 после runtime PR.

Clinical scores, weights, API queries и данные не меняются. На момент первой
записи UI implemented и локально проверен; CI, merge и deploy ещё не выполнены.
