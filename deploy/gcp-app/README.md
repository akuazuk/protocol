# deploy/gcp-app

Образ `protocol-gcp-app` (эпохи E1/E2).

```bash
docker build -f deploy/gcp-app/Dockerfile -t protocol-gcp-app .
```

Живой инвентарь E1 staging: [INVENTORY.md](INVENTORY.md).  
Startup VM: [startup-protocol-app.sh](startup-protocol-app.sh).  
Inbound score (без MIS): [score_inbound_day.sh](score_inbound_day.sh).  
План: `docs/plans/2026-08-07-by-home-gcp-llm-split-v1.md`.

```bash
# after GCS pull
bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --remote --limit 5
bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --remote
```
