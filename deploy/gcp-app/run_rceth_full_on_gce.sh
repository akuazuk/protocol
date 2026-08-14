#!/usr/bin/env bash
# Полный rceth sync на GCE: все действующие _s.pdf (RCETH_LIMIT=0), без лимита букв.
# Resume-safe (download skip existing). Watchdog подхватит last_job.env при падении.
#
#   bash deploy/gcp-app/run_rceth_full_on_gce.sh
# Parse later only:
#   RCETH_PARSE=0 bash deploy/gcp-app/run_rceth_full_on_gce.sh
#   RCETH_SKIP_CRAWL=1 RCETH_SKIP_DOWNLOAD=1 RCETH_PARSE=1 bash deploy/gcp-app/run_rceth_full_on_gce.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
CONTAINER="${RCETH_SYNC_CONTAINER:-protocol-web}"
LIMIT="${RCETH_LIMIT:-0}"
THROTTLE="${RCETH_THROTTLE:-0.6}"
# empty = all letters
MAX_LETTERS="${RCETH_MAX_LETTERS:-}"
PARSE="${RCETH_PARSE:-1}"
SKIP_CRAWL="${RCETH_SKIP_CRAWL:-0}"
SKIP_DOWNLOAD="${RCETH_SKIP_DOWNLOAD:-0}"
REMOTE_CODE="/var/data/rceth/_code"
DATA="/var/data/rceth"
PROTO="/opt/protocol"

echo "FULL rceth sync project=$PROJECT vm=$VM limit=$LIMIT (0=all) parse=$PARSE"

gcloud config set project "$PROJECT" --quiet >/dev/null

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p '$DATA/pdfs/instr' '$DATA/_sync' '$DATA/labels' '$REMOTE_CODE' '$PROTO/deploy/gcp-app'
sudo chown -R \"\$(whoami):\$(whoami)\" '$DATA'
mkdir -p '$REMOTE_CODE/clinical_knowledge' '$REMOTE_CODE/scripts'
"

gcloud compute scp --zone="$ZONE" --quiet --recurse \
  "$ROOT/clinical_knowledge/rceth_sync" \
  "$VM:$REMOTE_CODE/clinical_knowledge/"
gcloud compute scp --zone="$ZONE" --quiet \
  "$ROOT/scripts/rceth_sync_run.py" \
  "$VM:$REMOTE_CODE/scripts/rceth_sync_run.py"
gcloud compute scp --zone="$ZONE" --quiet \
  "$ROOT/deploy/gcp-app/rceth_sync_job.sh" \
  "$ROOT/deploy/gcp-app/rceth_sync_watchdog.sh" \
  "$VM:$REMOTE_CODE/scripts/"

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
if [[ ! -f $REMOTE_CODE/clinical_knowledge/__init__.py ]]; then
  printf '' > $REMOTE_CODE/clinical_knowledge/__init__.py
fi
docker cp '$REMOTE_CODE/clinical_knowledge/rceth_sync' '$CONTAINER:/app/clinical_knowledge/'
docker cp '$REMOTE_CODE/scripts/rceth_sync_run.py' '$CONTAINER:/app/scripts/rceth_sync_run.py'
chmod +x '$REMOTE_CODE/scripts/rceth_sync_job.sh' '$REMOTE_CODE/scripts/rceth_sync_watchdog.sh'
sudo cp -f '$REMOTE_CODE/scripts/rceth_sync_job.sh' '$PROTO/deploy/gcp-app/rceth_sync_job.sh'
sudo cp -f '$REMOTE_CODE/scripts/rceth_sync_watchdog.sh' '$PROTO/deploy/gcp-app/rceth_sync_watchdog.sh'
sudo chmod +x '$PROTO/deploy/gcp-app/rceth_sync_job.sh' '$PROTO/deploy/gcp-app/rceth_sync_watchdog.sh'

# refuse if already running
if docker top '$CONTAINER' 2>/dev/null | grep -q rceth_sync_run; then
  echo 'ERROR: rceth_sync already running in container' >&2
  exit 3
fi
if pgrep -f 'rceth_sync_job.sh' >/dev/null 2>&1; then
  echo 'ERROR: rceth_sync_job.sh already on host' >&2
  exit 3
fi

nohup env \
  RCETH_MODE=full \
  RCETH_LIMIT='$LIMIT' \
  RCETH_THROTTLE='$THROTTLE' \
  RCETH_MAX_LETTERS='$MAX_LETTERS' \
  RCETH_HTTP_TIMEOUT='${RCETH_HTTP_TIMEOUT:-30}' \
  RCETH_HTTP_RETRIES='${RCETH_HTTP_RETRIES:-3}' \
  RCETH_INSECURE_SSL=1 \
  RCETH_SYNC_CONTAINER='$CONTAINER' \
  RCETH_PARSE='$PARSE' \
  RCETH_SKIP_CRAWL='$SKIP_CRAWL' \
  RCETH_SKIP_DOWNLOAD='$SKIP_DOWNLOAD' \
  RCETH_DATA_ROOT='$DATA' \
  RCETH_PDF_MAX_BYTES='${RCETH_PDF_MAX_BYTES:-8388608}' \
  PROTOCOL_ROOT='$PROTO' \
  bash '$PROTO/deploy/gcp-app/rceth_sync_job.sh' \
  >'$DATA/_sync/full_launch.log' 2>&1 &
echo \"started host_pid=\$!\"
sleep 5
echo '--- status ---'
cat '$DATA/_sync/status.json' 2>/dev/null || echo 'status not yet'
echo '--- last_job.env ---'
cat '$DATA/_sync/last_job.env' 2>/dev/null || true
echo 'ETA: download ~1-5h for ~3.5-4k PDF; parse/OCR may take overnight+. Watch: bash deploy/gcp-app/watch_rceth_sync.sh'
"
