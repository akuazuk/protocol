# GCP inventory: protocol-home-e1 (E1 staging)

Создано 2026-08-07. Render DNS **не** переключали.

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
| Temp URL | `http://34.118.21.47:8000` |
| Container | `protocol-web` · image `protocol-gcp-app:staging` · restart unless-stopped |
| Smoke (2026-08-07) | `/health/live` ok · `/api/version` ok · `/api/methodist/mo/meta` ok |
| Data migrate | Render `/var/data/medical_exams` → GCE (234 MB); `fact_mo_case=97284` match |
| Firewall | `protocol-allow-web` · tcp 80/443/8000 · tag `protocol-app` |
| GCS inbound | `gs://protocol-home-e1-inbound` (`EUROPE-CENTRAL2`) |
| Startup | `deploy/gcp-app/startup-protocol-app.sh` (Docker + mount) |
| Redeploy | `bash deploy/gcp-app/deploy_to_gce.sh` (нужен локальный `.env`) |

## Ops

```bash
gcloud config set project protocol-home-e1
gcloud compute ssh protocol-app --zone=europe-central2-a
gcloud compute instances stop protocol-app --zone=europe-central2-a   # экономия
gcloud compute instances start protocol-app --zone=europe-central2-a
bash deploy/gcp-app/deploy_to_gce.sh
```

Следующее: migrate `medical_exams` с Render (B2 continue), HTTPS/temp host name, Mac → GCS extract.
