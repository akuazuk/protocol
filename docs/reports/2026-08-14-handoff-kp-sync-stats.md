# Handoff: расширенная статистика вкладки «Протоколы МЗ»

Дата: 2026-08-14

## Repo

- worktree: `/private/tmp/protocol-task-kp-sync-stats-pc1`
- branch: `cursor/kp-sync-stats-pc1`
- base: `origin/main` `b17df10`
- `BUILD_VERSION`: `2026-08-14-054846Z-kp-sync-stats`

## Сделано

Вкладка МО **Протоколы МЗ** показывает две шкалы дат (постановление МЗ и ночная сверка), сводку 7/30/90/YTD и диаграммы (ночи, месяцы, годы, рубрики). API `GET /api/methodist/mo/kp-sync` отдаёт `history`, `sync_periods`, `post_periods`, `by_year`, `by_month`, `by_slug`, `recent_posts`.

## Не сделано

- merge и deploy на GCE
- smoke вкладки на `https://protocol.kravira.by`

## Тесты

```bash
PYTHONPATH=. python3 -m pytest tests/test_kp_sync.py tests/test_mo_frontend_structure.py tests/test_mo_yesterday.py --noconftest
```

31 passed.

## Следующая команда после merge

```bash
GCE_OPS_USER=pavel SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh
```

Smoke: `/health/live` и `/api/version` на `protocol.kravira.by`, затем вкладка «Протоколы МЗ».

## Не трогать параллельно

- `clinical_knowledge/kp_sync/status.py`
- `frontend/web/shared/mo-app.js` (`loadKpSync`)
- `frontend/web/methodist/mis-kz-quality.html` (`page-kp-sync`)
