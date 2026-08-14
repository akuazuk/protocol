# Handoff: rceth sync status visible

- Repo: `akuazuk/protocol`
- Branch: `cursor/rceth-sync-status-visible-pc1`
- Worktree: `/private/tmp/protocol-task-rceth-sync-status-visible-pc1`
- Base: `origin/main` @ `cf71037`
- HEAD: `bdfc4e6`
- PR: https://github.com/akuazuk/protocol/pull/151
- `BUILD_VERSION`: `2026-08-14-100901Z-rceth-status-visible`

## Done

- Explained / fixed why «процесс не виден»: Notes card was static placeholder; live banner hidden unless `running`; `status.json` stuck as running after dead parse.
- `resolve_live_status` (dead PID / stale heartbeat → `interrupted`).
- Always-visible status card; clarified Notes copy.
- Parse skips PDF > 8MB by default.
- GCE `status.json` cleared to `interrupted` (ops).

## Not done

- Merge PR #151 + `deploy_to_gce.sh` + smoke on `protocol.kravira.by`.
- Re-run parse pilot after deploy.
- Step F findings UI.

## Next

```bash
# after merge:
bash deploy/gcp-app/deploy_to_gce.sh
bash deploy/gcp-app/run_rceth_sync_on_gce.sh  # or parse-only pilot
bash deploy/gcp-app/watch_rceth_sync.sh --once
```

Do not edit in parallel: `clinical_knowledge/rceth_sync/status.py`, `frontend/web/shared/mo-app.js` (rceth), `mis-kz-quality.html` rceth page.
