#!/usr/bin/env bash
# On GCE (or via gcloud compute ssh): sync GCS extract → local inbound/extract.
#
#   bash deploy/gcp-app/pull_inbound_from_gcs.sh
#   bash deploy/gcp-app/pull_inbound_from_gcs.sh --remote   # run on protocol-app via gcloud
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUCKET="${GCS_BUCKET:-gs://protocol-home-e1-inbound}"
PREFIX="${GCS_EXTRACT_PREFIX:-inbound/extract}"
# Local pull may use MO_DATA_ROOT; --remote always targets GCE disk path.
LOCAL_DEST="${MO_DATA_ROOT:-$ROOT/data/medical_exams}/inbound/extract"
GCE_DEST="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}/inbound/extract"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"

if [[ "${1:-}" == "--remote" ]]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null
  echo "Sync ${BUCKET}/${PREFIX}/ → ${VM}:${GCE_DEST}/"
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p '${GCE_DEST}'
tmp=\$(mktemp -d)
gcloud storage rsync '${BUCKET}/${PREFIX}/' \"\$tmp/\"
sudo cp -a \"\$tmp\"/. '${GCE_DEST}'/
sudo chmod -R a+rX '${GCE_DEST}'
rm -rf \"\$tmp\"
ls -la '${GCE_DEST}' | tail -20
"
  exit 0
fi

mkdir -p "$LOCAL_DEST"
gcloud config set project "$PROJECT" --quiet >/dev/null
echo "Sync ${BUCKET}/${PREFIX}/ → ${LOCAL_DEST}/"
gcloud storage rsync "${BUCKET}/${PREFIX}/" "${LOCAL_DEST}/"
ls -la "$LOCAL_DEST" | tail -20
