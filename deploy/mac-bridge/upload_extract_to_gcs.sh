#!/usr/bin/env bash
# Upload day extract (CSV+meta) from Mac/local inbound/extract → GCS.
#
#   bash deploy/mac-bridge/upload_extract_to_gcs.sh 2026-08-06
#   MO_DATA_ROOT=... GCS_BUCKET=gs://protocol-home-e1-inbound bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DAY="${1:?day YYYY-MM-DD}"
DATA_ROOT="${MO_DATA_ROOT:-$ROOT/data/medical_exams}"
BUCKET="${GCS_BUCKET:-gs://protocol-home-e1-inbound}"
PREFIX="${GCS_EXTRACT_PREFIX:-inbound/extract}"
LOCAL_DIR="${DATA_ROOT}/inbound/extract"
CSV="${LOCAL_DIR}/mo_${DAY}.csv"
META="${LOCAL_DIR}/mo_${DAY}.meta.json"

if [[ ! -f "$CSV" || ! -f "$META" ]]; then
  echo "ERROR: missing $CSV or $META" >&2
  echo "Run first: MO_DATA_ROOT=$DATA_ROOT PYTHONPATH=. python3 -m services.mis_bridge.extract_day --day $DAY --from-secure --run-host mac" >&2
  exit 2
fi

gcloud config set project "${GCP_PROJECT:-protocol-home-e1}" --quiet >/dev/null
DEST="${BUCKET}/${PREFIX}/"
echo "Uploading $DAY → ${DEST}"
gcloud storage cp "$CSV" "$META" "$DEST"
echo "OK"
gcloud storage ls "${DEST}mo_${DAY}.*"
