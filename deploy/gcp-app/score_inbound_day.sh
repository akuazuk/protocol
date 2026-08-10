#!/usr/bin/env bash
# Score one day from GCS/GCE inbound extract CSV (no MariaDB).
#
# On GCE host:
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --force
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --limit 5   # smoke
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --with-llm
#
# From Mac:
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --remote
#   bash deploy/gcp-app/score_inbound_day.sh 2026-08-06 --remote --limit 5
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DAY="${1:?YYYY-MM-DD}"
shift || true

FORCE=0
WITH_LLM=0
REMOTE=0
LIMIT="${MO_INBOUND_SCORE_LIMIT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --with-llm) WITH_LLM=1; shift ;;
    --remote) REMOTE=1; shift ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

CONTAINER="${CONTAINER:-protocol-web}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
Y="${DAY:0:4}"
M="${DAY:5:2}"

run_inside() {
  local limit_arg=""
  if [[ "${LIMIT}" != "0" && -n "${LIMIT}" ]]; then
    limit_arg="--limit ${LIMIT}"
  fi
  sudo docker exec \
    -e MO_DATA_ROOT="$DATA" \
    -e DAY="$DAY" -e Y="$Y" -e M="$M" \
    -e FORCE="$FORCE" -e LIMIT="$LIMIT" \
    -e MO_DAILY_WORKERS="${MO_DAILY_WORKERS:-2}" \
    "$CONTAINER" bash -lc '
set -euo pipefail
DATA="${MO_DATA_ROOT:-/var/data/medical_exams}"
IN="$DATA/inbound/extract/mo_${DAY}.csv"
META="$DATA/inbound/extract/mo_${DAY}.meta.json"
SECURE="$DATA/secure_cases/${Y}/${M}"
[[ -f "$IN" ]] || { echo "missing inbound CSV: $IN" >&2; exit 2; }
mkdir -p "$SECURE"
cp -f "$IN" "$SECURE/mo_${DAY}.csv"
if [[ -f "$META" ]]; then
  cp -f "$META" "$SECURE/mo_${DAY}.meta.json"
fi
if [[ "${FORCE}" == "1" ]]; then
  for s in cases.jsonl state.jsonl summary.json llm_queue.json; do
    rm -f "$SECURE/kz_l1_${DAY}_$s"
  done
fi
RESUME=(--resume)
[[ "${FORCE}" == "1" ]] && RESUME=()
LIMIT_ARGS=()
if [[ "${LIMIT}" != "0" && -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
echo "SCORE inbound day=$DAY force=$FORCE limit=${LIMIT:-0} workers=${MO_DAILY_WORKERS:-2}"
python scripts/run_mis_protocol_l1_batch.py \
  --csv "$SECURE/mo_${DAY}.csv" \
  --out-dir "$SECURE" \
  --month "$DAY" \
  --direct --deep-eval \
  "${RESUME[@]}" \
  "${LIMIT_ARGS[@]}" \
  --workers "${MO_DAILY_WORKERS:-2}"
python scripts/recompute_mo_days.py \
  --data-root "$DATA" \
  --first-date "$DAY" \
  --last-date "$DAY" \
  --warehouse "$DATA/warehouse/mo_analytics.sqlite"
if [[ -f /opt/protocol/scripts/mo_apply_scoring_profile_on_load.py || -f scripts/mo_apply_scoring_profile_on_load.py ]]; then
  MO_DATA_ROOT="$DATA" python scripts/mo_apply_scoring_profile_on_load.py \
    --data-root "$DATA" --wait || true
fi
echo SCORE_INBOUND_OK
'
}

if [[ "$REMOTE" == "1" ]]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null
  gcloud compute scp "$ROOT/deploy/gcp-app/score_inbound_day.sh" \
    "${VM}:/tmp/score_inbound_day.sh" --zone="$ZONE" --quiet
  EXTRA=()
  [[ "$FORCE" == "1" ]] && EXTRA+=(--force)
  [[ "$LIMIT" != "0" ]] && EXTRA+=(--limit "$LIMIT")
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
sudo cp /tmp/score_inbound_day.sh /opt/protocol/deploy/gcp-app/score_inbound_day.sh 2>/dev/null || true
sudo mkdir -p /opt/protocol/deploy/gcp-app
sudo cp /tmp/score_inbound_day.sh /opt/protocol/deploy/gcp-app/score_inbound_day.sh
sudo chmod +x /opt/protocol/deploy/gcp-app/score_inbound_day.sh
bash /opt/protocol/deploy/gcp-app/score_inbound_day.sh '$DAY' ${EXTRA[*]:-}
"
else
  run_inside
fi

if [[ "$WITH_LLM" == "1" ]]; then
  bash "$ROOT/deploy/gcp-llm/run_on_gce.sh" "$DAY" --foreground
fi
