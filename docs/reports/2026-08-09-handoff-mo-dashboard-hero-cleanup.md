# Handoff: MO dashboard hero cleanup (D1)

Дата: 2026-08-09
Branch: `cursor/mo-dashboard-hero-cleanup-pc1`
Worktree: `/private/tmp/protocol-task-mo-dashboard-hero-cleanup-pc1`
Plan: `docs/plans/2026-08-09-mo-dashboards-zones-first-v2.md`
PR: https://github.com/akuazuk/protocol/pull/90

## Сделано

- Меню слева: 6 видимых пунктов; `settings` оставлен `hidden` для admin accounts (#89).
- Период hero: плитки + тренд зон + «Куда смотреть» (+ Открыть); heatmap/Pareto/funnel/legacy trend скрыты.
- №55 (#87) только в `<details>` «Подробнее»; не в hero KPI-ряду.
- Сегодня: таблица дня + тренд зон; legacy индексы/поток скрыты.
- `hostActive` guards в `mo-app.js` - не рисуем hidden hosts.
- Пересобрано поверх `origin/main` после merge #87 и #89 (без конфликта с accounts API).

## Координация (другие вкладки)

| PR / работа | Статус | Наше действие |
|--|--|--|
| #87 №55 section-pack | merged | якоря + details UI сохранены; KPI №55 не в hero |
| #89 auth accounts | merged | auth/settings/`rag_server` accounts не трогали |
| #77 draft layers docs | open draft | не тащили UI; только docs index |
| Open dirty checkouts | другие вкладки | работа только в этом worktree |

## Тесты

`python3 tests/test_mo_dashboard_hero_cleanup.py`

## Не сделано

- D4 polish №55 aggregations (уже есть secondary UI от #87)
- D5 МКБ/clinical gaps в «Подробнее»
- GCE deploy после merge этого PR

## Одна следующая команда после merge

```bash
git fetch origin && git checkout -B deploy/mo-d1 origin/main
bash deploy/gcp-app/deploy_to_gce.sh
```
