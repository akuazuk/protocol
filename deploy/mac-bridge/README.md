# deploy/mac-bridge

Образ `protocol-mis-bridge` + контракт extract (эпоха E1 на Mac без Docker тоже ок).

## B4: extract → GCS

```bash
# package secure_cases CSV + upload (+ optional pull on GCE)
bash deploy/mac-bridge/extract_upload_day.sh 2026-08-06 --pull-gce

# pieces
PYTHONPATH=. python3 -m services.mis_bridge.extract_day --day 2026-08-06 --from-secure --run-host mac
bash deploy/mac-bridge/upload_extract_to_gcs.sh 2026-08-06
bash deploy/gcp-app/pull_inbound_from_gcs.sh --remote

# launchd mode (yesterday)
bash scripts/run_mo_daily_launchd.sh extract-upload-only
```

Bucket: `gs://protocol-home-e1-inbound/inbound/extract/`.  
Контракт: [extract-contract.md](extract-contract.md).

```bash
docker build -f deploy/mac-bridge/Dockerfile -t protocol-mis-bridge .
```
