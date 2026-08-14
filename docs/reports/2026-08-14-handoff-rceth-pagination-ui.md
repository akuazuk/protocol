# Handoff: rceth pagination + MO UI (2026-08-14)

- **repo:** akuazuk/protocol
- **branch:** `cursor/rceth-drug-labels-mo-v1-pc1`
- **worktree:** `/private/tmp/protocol-task-rceth-drug-labels-mo-pc1`
- **PR:** https://github.com/akuazuk/protocol/pull/146
- **HEAD:** see `git rev-parse HEAD` on branch
- **BUILD_VERSION:** `2026-08-14-075329Z-rceth-sync-ui-pagination`
- **merge/deploy:** нет (только task PR)

## Сделано

1. Пагинация Refbank: `IsPostBack=true` + `QueryStringFind` (буква «б»: 296/296).
2. `GET /api/methodist/mo/rceth-sync`, corpus-stats, page «Инструкции ЛС», live poll 2 с.
3. GCE job: `deploy/gcp-app/rceth_sync_job.sh` (пилот `RCETH_LIMIT=100`).
4. Тесты rceth + MO nav/auth.

## Не сделано

- Полный crawl/download на GCE `/var/data/rceth`.
- Identity merge / shadow findings.

## Следующая команда (на GCE, один writer)

```bash
RCETH_LIMIT=100 /opt/protocol/deploy/gcp-app/rceth_sync_job.sh
```

## Не трогать параллельно

`clinical_knowledge/rceth_sync/`, `frontend/web/shared/mo-app.js` (rceth page), `docs/plans/README.md`.
