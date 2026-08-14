#!/usr/bin/env bash
# Скопировать rceth_sync из текущего checkout на GCE и запустить пилот.
# Не делает deploy приложения. Данные только в /var/data/rceth.
#
#   bash deploy/gcp-app/run_rceth_sync_on_gce.sh
#   RCETH_LIMIT=50 bash deploy/gcp-app/run_rceth_sync_on_gce.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
CONTAINER="${RCETH_SYNC_CONTAINER:-protocol-web}"
LIMIT="${RCETH_LIMIT:-100}"
THROTTLE="${RCETH_THROTTLE:-0.6}"
MAX_LETTERS="${RCETH_MAX_LETTERS:-8}"
REMOTE_CODE="/var/data/rceth/_code"
DATA="/var/data/rceth"

echo "project=$PROJECT zone=$ZONE vm=$VM limit=$LIMIT max_letters=$MAX_LETTERS"

gcloud config set project "$PROJECT" --quiet >/dev/null

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p '$DATA/pdfs/instr' '$DATA/_sync' '$DATA/labels' '$REMOTE_CODE'
sudo chown -R \"\$(whoami):\$(whoami)\" '$DATA'
mkdir -p '$REMOTE_CODE/clinical_knowledge' '$REMOTE_CODE/scripts'
"

# sync package + CLI (text_extract берём из /app образа)
gcloud compute scp --zone="$ZONE" --quiet --recurse \
  "$ROOT/clinical_knowledge/rceth_sync" \
  "$VM:$REMOTE_CODE/clinical_knowledge/"
gcloud compute scp --zone="$ZONE" --quiet \
  "$ROOT/scripts/rceth_sync_run.py" \
  "$VM:$REMOTE_CODE/scripts/rceth_sync_run.py"
gcloud compute scp --zone="$ZONE" --quiet \
  "$ROOT/deploy/gcp-app/rceth_sync_job.sh" \
  "$VM:$REMOTE_CODE/scripts/rceth_sync_job.sh"

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
# минимальный пакет clinical_knowledge, чтобы импорт шёл с /app для остального
if [[ ! -f $REMOTE_CODE/clinical_knowledge/__init__.py ]]; then
  printf '' > $REMOTE_CODE/clinical_knowledge/__init__.py
fi
# docker cp в running app (эфемерно до recreate контейнера)
docker cp '$REMOTE_CODE/clinical_knowledge/rceth_sync' '$CONTAINER:/app/clinical_knowledge/'
docker cp '$REMOTE_CODE/scripts/rceth_sync_run.py' '$CONTAINER:/app/scripts/rceth_sync_run.py'
chmod +x '$REMOTE_CODE/scripts/rceth_sync_job.sh'

export PROTOCOL_ROOT=/opt/protocol
export RCETH_DATA_ROOT='$DATA'
export RCETH_LIMIT='$LIMIT'
export RCETH_THROTTLE='$THROTTLE'
export RCETH_MAX_LETTERS='$MAX_LETTERS'
export RCETH_INSECURE_SSL=1
export RCETH_SYNC_CONTAINER='$CONTAINER'
export RCETH_PARSE=1
nohup env RCETH_MAX_LETTERS='$MAX_LETTERS' RCETH_LIMIT='$LIMIT' RCETH_THROTTLE='$THROTTLE' \
  RCETH_HTTP_TIMEOUT='${RCETH_HTTP_TIMEOUT:-25}' RCETH_HTTP_RETRIES='${RCETH_HTTP_RETRIES:-3}' \
  RCETH_INSECURE_SSL=1 RCETH_SYNC_CONTAINER='$CONTAINER' RCETH_PARSE=1 RCETH_DATA_ROOT='$DATA' \
  bash '$REMOTE_CODE/scripts/rceth_sync_job.sh' \
  >'$DATA/_sync/pilot_launch.log' 2>&1 &
echo \"started pid=\$!\"
sleep 4
echo '--- status ---'
cat '$DATA/_sync/status.json' 2>/dev/null || echo 'status not yet'
echo '--- log ---'
tail -n 30 /var/data/medical_exams/logs/gce-rceth-sync.log 2>/dev/null || true
"
