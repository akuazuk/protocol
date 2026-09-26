## scripts/ops

Canonical entrypoints for multi-machine git/deploy operations.

During compatibility window, root-level scripts stay valid.
New automation and docs should prefer `scripts/ops/*`.

`render_deploy.sh` is the only entrypoint that talks to the Render API itself
(service settings, build logs, manual deploy and restart). It needs `RENDER_API_KEY`
in `.env`; everything else here works with git and the public prod URL only.

## Измерение МО Аналитики (план 2026-09-26, волна T)

- `mo_api_latency_probe.py` - тайминги всех горячих `/api/methodist/mo/*`
  (первый и повторный вызов, пороги плана, `--compare before after` для регресса).
  Токен из `METHODIST_TOKEN`, тела ответов не сохраняются.
- `mo_warehouse_profile.py` - агрегаты склада за месяц в JSON: покрытие по
  месяцам, распределения оценок и зон, состав замечаний, очередь догрузки,
  метрики раздела 7 плана. Без PHI; запускать на GCE read-only.
- `mo_ui_dom_audit.mjs` - Playwright-проход по всем экранам в 1024 и 1440:
  шапка, `details` закрытые/всего, скрытые пункты меню, ECharts, таблицы,
  overflow, шрифты, тайминги API; `--compare` показывает регресс между
  релизами. Скриншоты не делает.
