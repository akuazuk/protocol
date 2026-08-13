# Handoff: daily КП МЗ sync + вкладка МО — 2026-08-13

## Repo
- Branch: `cursor/minzdrav-kp-daily-sync-v1-pc1`
- Worktree: `/private/tmp/protocol-task-minzdrav-kp-daily-sync-pc1`
- PR: https://github.com/akuazuk/protocol/pull/142
- Plan: `docs/plans/2026-08-13-minzdrav-kp-daily-sync-v2.md`
- `BUILD_VERSION`: `2026-08-13-102737Z-kp-mo-tab`

## Done
- Daily crawl/diff/apply only for added/updated PDF; superseded не unlink.
- `run_pipeline --changed-only`: merge chunks/tables/cards по path (не full rewrite).
- Merge `protocol_catalog.jsonl` + ICD-профили; пустой extract не стирает старый ICD.
- Recency + skip `superseded` / `alias_of`; rehab/admin не clinical для plan-zone.
- Вкладка МО **Протоколы МЗ** (`/api/methodist/mo/kp-sync`, поле в `/api/corpus-stats`).
- Cron 01:00 UTC + retry 01:40 (stamp `.ok`); индексы пишутся в `/var/data/protocol_corpus` и монтируются в web.
- Тесты: `pytest tests/test_kp_sync.py tests/test_mo_frontend_structure.py tests/test_mo_expert_auth.py --noconftest`.

## Not done
- Catch-up ~61 missing PDF на GCE (после deploy: `KP_SYNC_FORCE=1 KP_SYNC_MAX_DOWNLOADS=80`).
- LLM summaries очередь; полный recompute июля; targeted re-score по ICD overlap (generation в JSON есть).
- Не коммитить git `index.csv` без PDF.

## Next after merge
```bash
GCE_OPS_USER=pavelkuzauka SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version
bash deploy/gcp-app/install_night_cron.sh --remote
```

Не параллелить: `clinical_knowledge/kp_sync/`, `deploy/gcp-app/night_kp_sync.sh`, `frontend/web/shared/mo-app.js` (вкладка kp-sync).
