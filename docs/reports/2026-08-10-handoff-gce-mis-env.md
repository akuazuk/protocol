# Handoff: GCE night MIS cron (Mac off)

Date: 2026-08-10  
Branch: `cursor/gce-mis-env-e2-pc1`  
PR: https://github.com/akuazuk/protocol/pull/121

## Done

- Night pipeline on GCE: `deploy/gcp-app/night_mis_pipeline.sh` (extract `mis_protocol`+`mis_data` → inbound → score → LLM bg).
- Cron (server = **UTC**): **02:00** main, **03:00** retry if status ≠ success.
- Day window: yesterday in **Europe/Minsk**.
- Retries on DB: `MO_DB_RETRIES=5` with exponential backoff.
- Mac `by.protocol.mo-daily*` launchd: **uninstalled** (no more Mac SQL pull).
- Install: `bash deploy/gcp-app/install_night_cron.sh --remote`

## Not done

- Secret Manager for password.
- Multi-night soak observation.

## Verify

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='crontab -l; tail -40 /var/data/medical_exams/logs/gce-night-main.log'
```

## Next safe command

```bash
MO_NIGHT_DAY=2026-08-09 bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh main
```
