# Handoff: MO runtime stabilization Phase A (2026-08-04)

Ветка: `codex/mo-runtime-stabilization-agent1-pc1`  
План: `docs/plans/2026-08-04-mo-runtime-stabilization-v1.md`

## Что сделано в коде (A1-A5)

1. **A1** - `assess_completeness`: очередь LLM не делает день `partial`, если coverage >= 99% и нет `scoring_errors`. Причины уходят в `advisory_reasons`.
2. **Reclassify** - `scripts/mo_reclassify_advisory_partial.py` переписывает report/public/state/warehouse без пересчёта оценок.
3. **A2** - `scripts/run_mo_daily_launchd.sh` грузит `METHODIST_TOKEN` из `ROOT/.env` / `PROTOCOL_ENV_FILE`.
4. **A3** - `publish_mo_to_render.py` возвращает код 3 при failed freshness / missing token; Telegram на fail.
5. **A4** - stale pid cleanup для `launchd-run.lock` и `pipeline.lock` (shell + `exclusive_lock`).
6. **A5** - health отдаёт `yesterday.{partial,reasons,advisory_reasons}`; UI «Вчера» показывает причины.

## Операционные шаги на Mac (A6)

Уже сделано 2026-08-04 с этой машины:

1. `mo_reclassify_advisory_partial.py --dates 2026-08-01 2026-08-02 2026-08-03` → days `passed`, LLM advisory.
2. `publish_mo_to_render.py` → remote `through=2026-08-03`, freshness **200**, `lag_days=1`.

После merge PR и Action deploy проверить UI «Вчера» (advisory banner) и `/api/methodist/mo/health` поле `yesterday`.

```bash
curl -fsS -H "X-Methodist-Token: $METHODIST_TOKEN" \
  https://protocol-bimy.onrender.com/api/methodist/mo/health | jq '.yesterday,.status,.reason_codes'
```

Не класть токен в LaunchAgents plist. После `manage_mo_daily_launchd.py install` wrapper сам читает `.env`.

## Не сделано

- Merge PR + Production Render release Action (код health/UI ещё не в `main`).
- Фаза B (Docker / cloud worker).
- Полный drain LLM-очереди (не обязателен при advisory-политике).
