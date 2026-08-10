# GCP inventory: protocol-home-e1 (E1 staging)

Создано 2026-08-07. С 2026-08-08: **primary UI** = этот GCE (`https://protocol.kravira.by`);
Render = backup (`https://protocol-bimy.onrender.com`). Автодеплоя после merge нет -
только `deploy_to_gce.sh`.

| Ресурс | Значение |
|--|--|
| Project | `protocol-home-e1` |
| Account | `aicoursesus@gmail.com` |
| Region / zone | `europe-central2` / `europe-central2-a` (Warsaw) |
| VM | `protocol-app` · `e2-standard-2` (2 vCPU, 8 GB) · Debian 12 |
| Boot disk | 20 GB pd-balanced |
| Data disk | `protocol-data` · 50 GB pd-balanced · mount `/var/data` |
| MO root on VM | `/var/data/medical_exams` (`inbound/extract`, warehouse, llm_*) |
| Static IP | `34.118.21.47` (`protocol-app-ip`) |
| Direct app | `http://34.118.21.47:8000` (Docker; keep for debug) |
| HTTPS | `https://protocol.kravira.by` → Caddy → `127.0.0.1:8000` |
| TLS | Caddy 2 + Let's Encrypt (CN=`protocol.kravira.by`, YE2; setup: `setup_https_caddy.sh --remote`) |
| DNS | hoster.by **A** `protocol.kravira.by` → `34.118.21.47` (CNAME снят; TTL 3600) |
| Container | `protocol-web` · image `protocol-gcp-app:staging` · restart unless-stopped |
| Smoke (2026-08-07) | `/health/live` ok · `/api/version` ok · `/api/methodist/mo/meta` ok |
| Data migrate | Render `/var/data/medical_exams` → GCE (234 MB); `fact_mo_case=97284` match |
| Firewall | `protocol-allow-web` · tcp 80/443/8000 · tag `protocol-app` |
| GCS inbound | `gs://protocol-home-e1-inbound` (`EUROPE-CENTRAL2`) |
| Startup | `deploy/gcp-app/startup-protocol-app.sh` (Docker + mount) |
| Redeploy | `bash deploy/gcp-app/deploy_to_gce.sh` (нужен локальный `.env` + sql_epam MIS) |
| MIS env | `/opt/protocol/.env.mis` ← `push_mis_env.sh` / deploy; smoke `mis_sql_smoke_on_gce.sh` |
| Night MIS | cron UTC `0 2` main + `0 3` retry + `15 3` check; `MO_DAILY_WORKERS=2`; skip unchanged sha |
| Night LLM | `bash deploy/gcp-llm/run_on_gce.sh <day>` (smoke: `--smoke`); also after night extract |
| Inbound GCS | `gs://protocol-home-e1-inbound/inbound/extract/` ← `extract_upload_day.sh` |
| Inbound on VM | `/var/data/medical_exams/inbound/extract/` ← `pull_inbound_from_gcs.sh --remote` |
| Score inbound | `bash deploy/gcp-app/score_inbound_day.sh <day> --remote` (no MIS) |
| HTTPS setup | `bash deploy/gcp-app/setup_https_caddy.sh --remote` |

## Ops

```bash
gcloud config set project protocol-home-e1
gcloud compute ssh protocol-app --zone=europe-central2-a
gcloud compute instances stop protocol-app --zone=europe-central2-a   # экономия
gcloud compute instances start protocol-app --zone=europe-central2-a
bash deploy/gcp-app/deploy_to_gce.sh
```

Smoke HTTPS (2026-08-07): `/health/live` + `/api/version` via `34.118.21.47` ok.
Следующее: Secret Manager; проверить 1–2 ночи GCE cron 02:00/03:00.
