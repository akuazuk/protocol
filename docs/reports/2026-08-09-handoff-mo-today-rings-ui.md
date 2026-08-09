# Handoff: Today rings + dynamics (wave 1)

Дата: 2026-08-09

## Repo / branch

| | |
|--|--|
| repo | `akuazuk/protocol` |
| worktree | `/private/tmp/protocol-task-mo-today-rings-ui-pc1` |
| branch | `cursor/mo-today-rings-ui-pc1` |
| base | `origin/main` @ `56b2d6bd` (#104) |
| HEAD | `bea527b3` |
| PR | https://github.com/akuazuk/protocol/pull/106 |
| BUILD_VERSION | `2026-08-09-101202Z-mo-today-rings-ui` |

## Сделано

- UI «Сегодня»: KPI strip, `#yesterday-score-rings` (4 doughnut), `#yesterday-score-dynamics` (зоны + №55).
- Данные колец/динамики из `/api/methodist/mo/score-dashboard` по `#period` (окно B).
- Таблица дня + attention tiles остаются на рабочем дне (окно A); подпись `yesterday-analytics-window`.
- Клики: сегмент зоны → documents `zone`+`zone_band`; №55 → `reg55_band`; точка динамики → custom day.
- Тесты UI обновлены; план `2026-08-09-mo-today-rings-dynamics-v1.md` - шаг волны 1 отмечен.

## Не сделано

- Merge / Render deploy / prod smoke.
- Волна 2: кольца на Период, МКБ мини-кольцо, stacked №55.

## Тесты

```bash
.venv/bin/python -m pytest \
  tests/test_mo_ui_phase2.py \
  tests/test_mo_dashboard_hero_cleanup.py \
  tests/test_mo_yesterday.py \
  tests/test_mo_frontend_structure.py \
  tests/test_mo_dashboard_nav_cleanup.py -q
```

Результат: ok.

## Deploy

Не запускался. После merge Action `Production Render release`; smoke `/methodist/mo` Сегодня + смена периода.

## Следующая команда

```bash
gh pr merge 106 --squash --delete-branch
```

(только release-координатор после review; затем смотреть Action и `/api/version`.)

## Не трогать параллельно

`frontend/web/shared/mo-app.js`, `mis-kz-quality.html`, `mo-ui.css` до merge #106.
