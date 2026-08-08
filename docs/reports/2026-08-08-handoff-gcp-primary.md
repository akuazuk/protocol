# Handoff: GCP primary + merge #46/#47 — 2026-08-08

## Repo
- `origin/main` @ `0252c824` (docs GCP primary #47)
- Prior: `e395f593` (#46 Caddy HTTPS)

## Done
- Merged PR #46 (Caddy HTTPS scripts)
- Merged PR #47: `AGENTS.md` + release rules - primary = `https://protocol.kravira.by`, Render backup
- Manual deploy: `deploy_to_gce.sh` → version `2026-08-08-055439Z-gcp-primary-rules`
- Smoke: `/health/live` ok, `/api/methodist/mo/meta` ok

## Not done / decisions
- **No GCE auto-deploy Action yet** (too early: need SA in GitHub Secrets + 2–3 stable nights)
- Render still receives Action deploys as backup - do not delete
- Mac launchd still not forced to extract-upload-only
- Data gap: GCE last full day 2026-08-06

## Next
```bash
# after future merges that touch GCE app:
bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version
```
