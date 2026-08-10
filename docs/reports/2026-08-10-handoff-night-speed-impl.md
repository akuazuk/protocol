# Handoff: night speed + skip + alerts on GCE

Date: 2026-08-10  
Branch: `cursor/mo-night-speed-impl-v1-pc1`  
Base: includes GCE MIS night from `cursor/gce-mis-env-e2-pc1`

## Deployed on VM

- Cron: 02:00 main, 03:00 retry, **03:15 check** (`MO_DAILY_WORKERS=2`)
- Scripts under `/opt/protocol/deploy/gcp-app/`

## Smoke 2026-08-09

| Test | Result |
|--|--|
| main workers=2 | success, `workers=2` in score log, sha written |
| main again | `unchanged_skip_score`, `skipped_score=true` (~34s = re-extract only) |
| retry | rc=0 already success |
| check + fake fail | `ALERT_NEEDED`, check_rc=2; status restored |

## Next

Merge PR; optional Telegram keys in `.env.gcp-staging` if not set.
