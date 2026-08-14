#!/usr/bin/env bash
# Live-мониторинг rceth sync на GCE (status.json + хвост лога).
#
#   bash deploy/gcp-app/watch_rceth_sync.sh          # каждые 5 с
#   bash deploy/gcp-app/watch_rceth_sync.sh --once   # один снимок
set -euo pipefail

PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

gcloud config set project "$PROJECT" --quiet >/dev/null

snapshot() {
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command='
STATUS=/var/data/rceth/_sync/status.json
LOG=/var/data/medical_exams/logs/gce-rceth-sync.log
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
if [ -f "$STATUS" ]; then python3 -c "import json;d=json.load(open(\"$STATUS\"));p=d.get(\"progress\") or {};print(\"status=%s phase=%s done=%s/%s\"%(d.get(\"status\"),d.get(\"phase\"),p.get(\"done\"),p.get(\"total\")));print(\"message=%s current=%s errors=%s\"%(d.get(\"message\"),d.get(\"current_reg_id\"),d.get(\"errors\")));print(\"updated_at=%s\"%(d.get(\"updated_at\"),))"; else echo "status.json: missing"; fi
echo -n "pdfs="; ls /var/data/rceth/pdfs/instr 2>/dev/null | wc -l
echo -n "manifest_lines="; (test -f /var/data/rceth/manifest.jsonl && wc -l < /var/data/rceth/manifest.jsonl) || echo 0
echo -n "labels="; ls /var/data/rceth/labels 2>/dev/null | wc -l
echo -n "process="; docker top protocol-web 2>/dev/null | grep -c rceth_sync || echo 0
echo "-- log --"; tail -n 10 "$LOG" 2>/dev/null || echo "(no log)"
'
}

if [[ "$ONCE" == "1" ]]; then
  snapshot
  exit 0
fi

echo "watching GCE rceth (Ctrl+C to stop)"
echo "after app deploy: МО Аналитика → Инструкции ЛС (poll /api/methodist/mo/rceth-sync)"
while true; do
  snapshot || true
  sleep 5
done
