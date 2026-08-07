#!/usr/bin/env bash
# B4 one-shot: package day CSV (from secure_cases or --from-csv) → GCS.
#
#   bash deploy/mac-bridge/extract_upload_day.sh 2026-08-06
#   bash deploy/mac-bridge/extract_upload_day.sh 2026-08-06 --from-csv /path/mo.csv
#   bash deploy/mac-bridge/extract_upload_day.sh 2026-08-06 --pull-gce
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DAY="${1:?day YYYY-MM-DD}"
shift || true
FROM_CSV=""
PULL_GCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-csv) FROM_CSV="${2:?}"; shift 2 ;;
    --pull-gce) PULL_GCE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

export MO_DATA_ROOT="${MO_DATA_ROOT:-$ROOT/data/medical_exams}"
export RUN_HOST="${RUN_HOST:-mac}"

ARGS=(--day "$DAY" --run-host "$RUN_HOST")
if [[ -n "$FROM_CSV" ]]; then
  ARGS+=(--from-csv "$FROM_CSV")
else
  ARGS+=(--from-secure)
fi

echo "[1/3] package extract"
PYTHONPATH=. python3 -m services.mis_bridge.extract_day "${ARGS[@]}"

echo "[2/3] upload GCS"
bash "$ROOT/deploy/mac-bridge/upload_extract_to_gcs.sh" "$DAY"

if [[ "$PULL_GCE" == "1" ]]; then
  echo "[3/3] pull on GCE"
  bash "$ROOT/deploy/gcp-app/pull_inbound_from_gcs.sh" --remote
else
  echo "[3/3] skip GCE pull (pass --pull-gce)"
fi
echo DONE
