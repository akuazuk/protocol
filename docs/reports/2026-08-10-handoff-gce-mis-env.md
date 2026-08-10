# Handoff: E2 MIS env on GCE

Date: 2026-08-10  
Branch: `cursor/gce-mis-env-e2-pc1`  
Worktree: `/private/tmp/protocol-task-gce-mis-env-pc1`

## Done

- Canonical MIS secrets on GCE: `/opt/protocol/.env.mis` (chmod 600).
- Keys present: `KRAVIRA_DB_*`, `MIS_DB_*_TIMEOUT`, `RUN_HOST=gcp` (password not logged).
- Smoke from VM: `SQL_OK`, `mis_protocol_max_date=2026-08-09`.
- `export_mis_protocol_month.py` env-first (Mac `sql_epam` only fallback).
- `deploy_to_gce.sh` uploads `.env.mis` when local password available.
- Helpers: `push_mis_env.sh`, `mis_sql_smoke_on_gce.sh`.
- Docs/plan/rules: E2 C1 done, C2 env-file in progress (Secret Manager next).

## Not done

- Secret Manager migration for MIS password.
- GCE nightly MIS extract job (mis_bridge on VM); Mac launchd still possible fallback.
- Full `deploy_to_gce.sh` not required for this secret-only change.

## Next safe command

```bash
bash deploy/gcp-app/mis_sql_smoke_on_gce.sh
```

## Do not touch in parallel

- `deploy/gcp-app/deploy_to_gce.sh`, `ENV-MIGRATION.md`, `/opt/protocol/.env.mis` on VM.
