# Handoff: MO Analytics visual refresh (A+B)

- repo: `akuazuk/protocol`
- branch: `cursor/mo-ui-visual-refresh-pc1`
- worktree: `/private/tmp/protocol-task-mo-ui-visual-refresh-pc1`
- plan: `docs/plans/2026-08-09-mo-analytics-visual-refresh-v1.md`
- BUILD_VERSION: `2026-08-09-091817Z-mo-ui-visual-refresh`

## Done

- Owner decisions locked: color умеренно+, search+chips+col dropdowns, Avenir scale, no CSV yet
- Tokens: brighter zones/severity + type roles
- `attachTableChrome` / `enhanceTablesIn` on Сегодня / Период / Очередь / Документы / Врачи tables
- Tests in `test_mo_ui_phase2.py`

## Not done

- Wave C: charts/KPI color polish
- CSV export on Сегодня/Врачи
- Merge/deploy

## Parallel

Touch: `mo-tokens.css`, `mo-ui.css`, `mo-app.js`, `docs/plans/*`
