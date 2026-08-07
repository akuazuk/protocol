#!/usr/bin/env bash
# Start night LLM + action-judge on GCE protocol-app (not Render SSH, not Mac).
#
# Usage:
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-06
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --foreground
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --smoke   # grade --limit 1 only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${GCP_PROJECT:-protocol-home-e1}"
ZONE="${GCP_ZONE:-europe-central2-a}"
VM="${GCP_VM:-protocol-app}"
CONTAINER="${CONTAINER:-protocol-web}"
DATA="${MO_DATA_ROOT:-/var/data/medical_exams}"

FIRST="${1:?first date YYYY-MM-DD}"
LAST="$FIRST"
MODE=""
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground|--smoke) MODE="$1" ;;
    20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]) LAST="$1" ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

gcloud config set project "$PROJECT" --quiet >/dev/null
STATUS="$(gcloud compute instances describe "$VM" --zone="$ZONE" --format='get(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "Starting VM $VM ..."
  gcloud compute instances start "$VM" --zone="$ZONE" --quiet
  sleep 20
fi

echo "[1/3] sync runner + grade scripts to VM"
gcloud compute scp \
  "$ROOT/scripts/mo_llm_range_runner.sh" \
  "$ROOT/scripts/grade_kz_llm.py" \
  "$ROOT/scripts/run_mo_action_queue_llm_judge.py" \
  "$ROOT/scripts/recompute_mo_days.py" \
  "${VM}:/tmp/" --zone="$ZONE" --quiet

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p /opt/protocol/scripts '${DATA}/logs'
sudo cp /tmp/mo_llm_range_runner.sh /tmp/grade_kz_llm.py \
  /tmp/run_mo_action_queue_llm_judge.py /tmp/recompute_mo_days.py \
  /opt/protocol/scripts/
sudo chmod +x /opt/protocol/scripts/mo_llm_range_runner.sh
if sudo docker ps --format '{{.Names}}' | grep -qx '${CONTAINER}'; then
  sudo docker cp /opt/protocol/scripts/mo_llm_range_runner.sh '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/grade_kz_llm.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/run_mo_action_queue_llm_judge.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/recompute_mo_days.py '${CONTAINER}':/app/scripts/
fi
"

Y="${FIRST:0:4}"
M="${FIRST:5:2}"

if [[ "$MODE" == "--smoke" ]]; then
  echo "[2/3] smoke: grade --limit 1 to /tmp (live Gemini from GCE, no resume)"
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo docker exec '${CONTAINER}' python scripts/grade_kz_llm.py \
  --cases '${DATA}/secure_cases/${Y}/${M}/kz_l1_${FIRST}_cases.jsonl' \
  --queue '${DATA}/secure_cases/${Y}/${M}/kz_l1_${FIRST}_llm_queue.json' \
  --out /tmp/gcp_smoke_grades_${FIRST}.jsonl \
  --warehouse '${DATA}/warehouse/mo_analytics.sqlite' \
  --run-id 'gcp-smoke-${FIRST}' \
  --limit 1 --escalate
sudo docker exec '${CONTAINER}' wc -l /tmp/gcp_smoke_grades_${FIRST}.jsonl
echo SMOKE_GRADE_OK
"
  echo "[3/3] done smoke"
  exit 0
fi

echo "[2/3] start range runner FIRST=${FIRST} LAST=${LAST} MODE=${MODE:-background}"
if [[ "$MODE" == "--foreground" ]]; then
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
if sudo docker exec '${CONTAINER}' pgrep -f 'python .*grade_kz_llm[.]py' >/dev/null 2>&1; then
  echo ALREADY_RUNNING grade_kz_llm
  sudo docker exec '${CONTAINER}' pgrep -af 'grade_kz_llm' | head -3
  exit 0
fi
sudo docker exec \
  -e FIRST='${FIRST}' -e LAST='${LAST}' \
  -e SRC_ROOT=/app -e DATA='${DATA}' \
  -e PYTHON=python -e RUN_HOST=gcp -e RUN_ID_PREFIX=gcp-llm \
  -e MO_ACTION_JUDGE_LIMIT='${MO_ACTION_JUDGE_LIMIT:-0}' \
  '${CONTAINER}' bash /app/scripts/mo_llm_range_runner.sh
"
else
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
if sudo docker exec '${CONTAINER}' pgrep -f 'python .*grade_kz_llm[.]py' >/dev/null 2>&1; then
  echo ALREADY_RUNNING grade_kz_llm
  sudo docker exec '${CONTAINER}' pgrep -af 'grade_kz_llm' | head -3
  exit 0
fi
sudo docker exec -d \
  -e FIRST='${FIRST}' -e LAST='${LAST}' \
  -e SRC_ROOT=/app -e DATA='${DATA}' \
  -e PYTHON=python -e RUN_HOST=gcp -e RUN_ID_PREFIX=gcp-llm \
  -e MO_ACTION_JUDGE_LIMIT='${MO_ACTION_JUDGE_LIMIT:-0}' \
  '${CONTAINER}' bash /app/scripts/mo_llm_range_runner.sh
sleep 2
sudo docker exec '${CONTAINER}' pgrep -af 'mo_llm_range_runner|grade_kz_llm' | head -5 || true
echo LOG=${DATA}/logs/mo_llm_backfill_${FIRST}_${LAST}.log
"
fi
echo "[3/3] started on GCE ${VM}"
echo "Tail: gcloud compute ssh ${VM} --zone=${ZONE} --command=\"sudo tail -f ${DATA}/logs/mo_llm_backfill_${FIRST}_${LAST}.log\""
