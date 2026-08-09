#!/usr/bin/env bash
# Start night LLM + action-judge on GCE protocol-app (not Render SSH, not Mac).
#
# Usage:
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-06
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --foreground
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06 --smoke   # grade --limit 1 only
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-08 --calibration-smoke
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-08 --calibration-pilot
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-08 --calibration-methodist-pack
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-01 2026-08-08 --calibration-agent-proxy
#   bash deploy/gcp-llm/run_on_gce.sh 2026-07-26 2026-07-31 --calibration-confirmatory-proxy
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
    --foreground|--smoke|--calibration-smoke|--calibration-pilot|--calibration-methodist-pack|--calibration-agent-proxy|--calibration-confirmatory-proxy) MODE="$1" ;;
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
  "$ROOT/scripts/run_mo_icd_llm_review.py" \
  "$ROOT/scripts/recompute_mo_days.py" \
  "$ROOT/scripts/build_mo_score_calibration_sample.py" \
  "$ROOT/scripts/run_mo_calibration_blind_judge.py" \
  "$ROOT/scripts/eval_mo_score_calibration.py" \
  "$ROOT/scripts/eval_mo_score_agent_proxy.py" \
  "$ROOT/scripts/select_mo_calibration_provisional.py" \
  "$ROOT/scripts/build_mo_calibration_methodist_pack.py" \
  "$ROOT/clinical_knowledge/mo_icd_llm_review.py" \
  "$ROOT/clinical_knowledge/mo_dx_evidence_score.py" \
  "$ROOT/clinical_knowledge/mo_plan_protocol_score.py" \
  "${VM}:/tmp/" --zone="$ZONE" --quiet

gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p /opt/protocol/scripts /opt/protocol/clinical_knowledge '${DATA}/logs'
sudo cp /tmp/mo_llm_range_runner.sh /tmp/grade_kz_llm.py \
  /tmp/run_mo_action_queue_llm_judge.py /tmp/run_mo_icd_llm_review.py \
  /tmp/recompute_mo_days.py /tmp/build_mo_score_calibration_sample.py \
  /tmp/run_mo_calibration_blind_judge.py /tmp/eval_mo_score_calibration.py \
  /tmp/eval_mo_score_agent_proxy.py /tmp/select_mo_calibration_provisional.py \
  /tmp/build_mo_calibration_methodist_pack.py \
  /opt/protocol/scripts/
sudo cp /tmp/mo_icd_llm_review.py /tmp/mo_dx_evidence_score.py \
  /tmp/mo_plan_protocol_score.py /opt/protocol/clinical_knowledge/
sudo chmod +x /opt/protocol/scripts/mo_llm_range_runner.sh
if sudo docker ps --format '{{.Names}}' | grep -qx '${CONTAINER}'; then
  sudo docker cp /opt/protocol/scripts/mo_llm_range_runner.sh '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/grade_kz_llm.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/run_mo_action_queue_llm_judge.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/run_mo_icd_llm_review.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/recompute_mo_days.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/build_mo_score_calibration_sample.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/run_mo_calibration_blind_judge.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/eval_mo_score_calibration.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/eval_mo_score_agent_proxy.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/select_mo_calibration_provisional.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/scripts/build_mo_calibration_methodist_pack.py '${CONTAINER}':/app/scripts/
  sudo docker cp /opt/protocol/clinical_knowledge/mo_icd_llm_review.py '${CONTAINER}':/app/clinical_knowledge/
  sudo docker cp /opt/protocol/clinical_knowledge/mo_dx_evidence_score.py '${CONTAINER}':/app/clinical_knowledge/
  sudo docker cp /opt/protocol/clinical_knowledge/mo_plan_protocol_score.py '${CONTAINER}':/app/clinical_knowledge/
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

if [[ "$MODE" == "--calibration-confirmatory-proxy" ]]; then
  CALIBRATION_DIR="${DATA}/calibration/mo-score-v3-confirmatory-${FIRST}-${LAST}"
  PILOT_MANIFEST="${DATA}/calibration/mo-score-v3-2026-08-01-2026-08-08/secret/secret_manifest.jsonl"
  TARGET_N="${MO_CALIBRATION_CONFIRMATORY_N:-100}"
  SEED="${MO_CALIBRATION_CONFIRMATORY_SEED:-43}"
  BLIND_MODEL="${MO_CALIBRATION_MODEL:-gemini-3.6-flash}"
  PROXY_MODEL="${MO_CALIBRATION_PROXY_MODEL:-gemini-3.1-pro-preview}"
  BLIND_OUT="${CALIBRATION_DIR}/secret/blind_confirmatory.jsonl"
  BLIND_SUMMARY="${CALIBRATION_DIR}/confirmatory_blind_summary.json"
  PROXY_OUT="${CALIBRATION_DIR}/secret/agent_proxy_confirmatory.jsonl"
  PROXY_SUMMARY="${CALIBRATION_DIR}/secret/agent_proxy_confirmatory_summary.json"
  PROXY_EVAL="${CALIBRATION_DIR}/agent_proxy_eval_summary.json"
  PROVISIONAL_OUT="${CALIBRATION_DIR}/provisional_methodology.json"
  Y1="${FIRST:0:4}"
  M1="${FIRST:5:2}"
  echo "[2/3] calibration C9A: confirmatory ${TARGET_N} cases + blind + proxy"
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p '${CALIBRATION_DIR}'
sudo docker exec \
  -e MO_LLM_EXECUTION_HOST=gce -e RUN_HOST=gcp \
  '${CONTAINER}' bash -lc \"
set -euo pipefail
cd /app
test -f '${PILOT_MANIFEST}'
python scripts/build_mo_score_calibration_sample.py \\
  --cases ${DATA}/secure_cases/${Y1}/${M1}/kz_l1_${Y1}-${M1}-??_cases.jsonl \\
  --clinical-csv ${DATA}/secure_cases/${Y1}/${M1}/mis_protocol_${Y1}-${M1}.csv \\
  --warehouse ${DATA}/warehouse/mo_analytics.sqlite \\
  --secret-dir '${CALIBRATION_DIR}/secret' \\
  --public-manifest '${CALIBRATION_DIR}/public_manifest.json' \\
  --date-from '${FIRST}' --date-to '${LAST}' \\
  --target-n '${TARGET_N}' --seed '${SEED}' --no-sentinel \\
  --exclude-manifest '${PILOT_MANIFEST}'
python -c 'import json; m=json.load(open(\\\"${CALIBRATION_DIR}/public_manifest.json\\\")); assert m[\\\"audit\\\"][\\\"passed\\\"]; assert m[\\\"selected_n\\\"]==int(\\\"${TARGET_N}\\\"); print(\\\"CONFIRMATORY_SAMPLE_OK\\\", m[\\\"selected_n\\\"])'
python scripts/eval_mo_score_calibration.py \\
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \\
  --snapshot '${CALIBRATION_DIR}/secret/engine_snapshot.jsonl' \\
  --replay '${CALIBRATION_DIR}/secret/engine_replay.jsonl' \\
  --out '${CALIBRATION_DIR}/replay_drift_summary.json'
python scripts/run_mo_calibration_blind_judge.py \\
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \\
  --manifest '${CALIBRATION_DIR}/secret/secret_manifest.jsonl' \\
  --out '${BLIND_OUT}' \\
  --summary-out '${BLIND_SUMMARY}' \\
  --limit 0 --passes 1 --model '${BLIND_MODEL}' --require-route-coverage --resume || true
python -c 'import json; value=json.load(open(\\\"${BLIND_SUMMARY}\\\")); assert value[\\\"parse_success_n\\\"]>=int(\\\"${TARGET_N}\\\")-2; assert value[\\\"leakage_failure_n\\\"]==0; assert value[\\\"geo_error_n\\\"]==0; print(\\\"CONFIRMATORY_BLIND_OK\\\")'
python scripts/run_mo_calibration_blind_judge.py \\
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \\
  --manifest '${CALIBRATION_DIR}/secret/secret_manifest.jsonl' \\
  --out '${PROXY_OUT}' \\
  --summary-out '${PROXY_SUMMARY}' \\
  --limit 0 --passes 1 --model '${PROXY_MODEL}' --require-route-coverage --resume || true
python -c 'import json; value=json.load(open(\\\"${PROXY_SUMMARY}\\\")); assert value[\\\"parse_success_n\\\"]>=int(\\\"${TARGET_N}\\\")-2; assert value[\\\"leakage_failure_n\\\"]==0; assert value[\\\"geo_error_n\\\"]==0; print(\\\"CONFIRMATORY_PROXY_OK\\\")'
python scripts/eval_mo_score_agent_proxy.py \\
  --snapshot '${CALIBRATION_DIR}/secret/engine_snapshot.jsonl' \\
  --replay '${CALIBRATION_DIR}/secret/engine_replay.jsonl' \\
  --blind '${BLIND_OUT}' \\
  --proxy '${PROXY_OUT}' \\
  --out '${PROXY_EVAL}' \\
  --bootstrap-iterations 2000 --seed 43
python scripts/select_mo_calibration_provisional.py \\
  --proxy-eval '${PROXY_EVAL}' \\
  --out '${PROVISIONAL_OUT}'
python -c 'import json; value=json.load(open(\\\"${PROVISIONAL_OUT}\\\")); assert value[\\\"production_rollout\\\"][\\\"allowed\\\"] is False; print(json.dumps(value[\\\"shadow_recommendation\\\"], ensure_ascii=False))'
\"
"
  echo "[3/3] done confirmatory proxy on GCE ${VM}"
  echo "SECRET_DIR=${CALIBRATION_DIR}/secret"
  echo "PUBLIC_AGGREGATE=${PROXY_EVAL}"
  echo "PROVISIONAL=${PROVISIONAL_OUT}"
  exit 0
fi

if [[ "$MODE" == "--calibration-agent-proxy" ]]; then
  CALIBRATION_DIR="${DATA}/calibration/mo-score-v3-${FIRST}-${LAST}"
  PROXY_MODEL="${MO_CALIBRATION_PROXY_MODEL:-gemini-3.1-pro-preview}"
  PROXY_OUT="${CALIBRATION_DIR}/secret/agent_proxy_gemini31pro.jsonl"
  PROXY_SUMMARY="${CALIBRATION_DIR}/secret/agent_proxy_gemini31pro_summary.json"
  PROXY_EVAL="${CALIBRATION_DIR}/agent_proxy_eval_summary.json"
  echo "[2/3] calibration C6A/C7A: independent AI proxy + aggregate comparison"
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo docker exec \
  -e MO_LLM_EXECUTION_HOST=gce -e RUN_HOST=gcp \
  '${CONTAINER}' bash -lc \"
set -euo pipefail
cd /app
python -c 'import hashlib,json,pathlib; from scripts.build_mo_score_calibration_sample import arm_d_fingerprint; root=pathlib.Path(\\\"${CALIBRATION_DIR}\\\"); m=json.load(open(root/\\\"public_manifest.json\\\")); assert m[\\\"audit\\\"][\\\"passed\\\"]; assert arm_d_fingerprint()[\\\"fingerprint\\\"]==m[\\\"arm_d_fingerprint\\\"][\\\"fingerprint\\\"]; expected=m[\\\"secret_artifact_hashes\\\"]; files=(\\\"secret_cases.jsonl\\\",\\\"secret_manifest.jsonl\\\",\\\"engine_snapshot.jsonl\\\",\\\"engine_replay.jsonl\\\"); assert all(hashlib.sha256((root/\\\"secret\\\"/name).read_bytes()).hexdigest()==expected[name] for name in files); print(\\\"FROZEN_SAMPLE_HASH_OK\\\")'
python scripts/run_mo_calibration_blind_judge.py \
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \
  --manifest '${CALIBRATION_DIR}/secret/secret_manifest.jsonl' \
  --out '${PROXY_OUT}' \
  --summary-out '${PROXY_SUMMARY}' \
  --limit 30 --passes 1 --model '${PROXY_MODEL}' --require-route-coverage --resume || true
python -c 'import json; value=json.load(open(\\\"${PROXY_SUMMARY}\\\")); assert value[\\\"parse_success_n\\\"]>=29 and value[\\\"error_n\\\"]<=1 and value[\\\"leakage_failure_n\\\"]==0 and value[\\\"geo_error_n\\\"]==0 and value[\\\"route_coverage_passed\\\"]; print(\\\"PROXY_EXPLORATORY_GATE_OK\\\")'
python scripts/eval_mo_score_agent_proxy.py \
  --snapshot '${CALIBRATION_DIR}/secret/engine_snapshot.jsonl' \
  --replay '${CALIBRATION_DIR}/secret/engine_replay.jsonl' \
  --blind '${CALIBRATION_DIR}/secret/blind_pilot.jsonl' \
  --proxy '${PROXY_OUT}' \
  --out '${PROXY_EVAL}' \
  --bootstrap-iterations 2000 --seed 42
python -c 'import json; value=json.load(open(\\\"${PROXY_EVAL}\\\")); assert not value[\\\"limitations\\\"][\\\"proxy_is_human_gold\\\"]; assert not value[\\\"limitations\\\"][\\\"production_decision_allowed\\\"]; assert value[\\\"phi_check\\\"][\\\"contains_clinical_text\\\"] is False; assert value[\\\"endpoints\\\"][\\\"dx\\\"][\\\"proxy_payload_n\\\"]>=29 and value[\\\"endpoints\\\"][\\\"plan\\\"][\\\"proxy_payload_n\\\"]>=29; print(json.dumps({\\\"analysis\\\":value[\\\"analysis\\\"],\\\"proxy_models\\\":value[\\\"proxy_models\\\"],\\\"run_quality\\\":value[\\\"proxy_run_quality\\\"],\\\"dx\\\":{k:value[\\\"endpoints\\\"][\\\"dx\\\"][k] for k in (\\\"proxy_payload_n\\\",\\\"proxy_labeled_n\\\",\\\"proxy_bad_n\\\",\\\"candidate_n\\\")},\\\"plan\\\":{k:value[\\\"endpoints\\\"][\\\"plan\\\"][k] for k in (\\\"proxy_payload_n\\\",\\\"proxy_labeled_n\\\",\\\"proxy_bad_n\\\",\\\"candidate_n\\\")}}, ensure_ascii=False))'
\"
"
  echo "[3/3] done calibration agent proxy on GCE ${VM}"
  echo "SECRET_PROXY=${PROXY_OUT}"
  echo "PUBLIC_AGGREGATE=${PROXY_EVAL}"
  exit 0
fi

if [[ "$MODE" == "--calibration-methodist-pack" ]]; then
  CALIBRATION_DIR="${DATA}/calibration/mo-score-v3-${FIRST}-${LAST}"
  echo "[2/3] calibration C6: build blinded methodist review pack"
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo docker exec '${CONTAINER}' bash -lc \"
set -euo pipefail
cd /app
python scripts/build_mo_calibration_methodist_pack.py \
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \
  --pilot '${CALIBRATION_DIR}/secret/blind_pilot.jsonl' \
  --secret-out-dir '${CALIBRATION_DIR}/secret/methodist' \
  --public-status '${CALIBRATION_DIR}/methodist_status.json'
\"
"
  echo "[3/3] done calibration methodist pack on GCE ${VM}"
  echo "SECRET_PACK=${CALIBRATION_DIR}/secret/methodist"
  echo "PUBLIC_STATUS=${CALIBRATION_DIR}/methodist_status.json"
  exit 0
fi

if [[ "$MODE" == "--calibration-smoke" || "$MODE" == "--calibration-pilot" ]]; then
  CALIBRATION_DIR="${DATA}/calibration/mo-score-v3-${FIRST}-${LAST}"
  if [[ "$MODE" == "--calibration-pilot" ]]; then
    JUDGE_OUT="${CALIBRATION_DIR}/secret/blind_pilot.jsonl"
    SUMMARY_OUT="${CALIBRATION_DIR}/pilot_summary.json"
    JUDGE_ARGS="--limit 0 --passes 2 --require-route-coverage --adjudicate-disagreements --resume"
    echo "[2/3] calibration C5: frozen sample, 30 cases x 2 + adjudication"
  else
    JUDGE_OUT="${CALIBRATION_DIR}/secret/blind_smoke.jsonl"
    SUMMARY_OUT="${CALIBRATION_DIR}/smoke_summary.json"
    JUDGE_ARGS="--limit 5 --passes 2 --require-route-coverage"
    echo "[2/3] calibration C0-C4: frozen sample, replay, 5 cases x 2 passes"
  fi
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="
set -euo pipefail
sudo mkdir -p '${CALIBRATION_DIR}'
sudo docker exec \
  -e MO_LLM_EXECUTION_HOST=gce -e RUN_HOST=gcp \
  '${CONTAINER}' bash -lc \"
set -euo pipefail
if [[ '${MODE}' == '--calibration-pilot' && -f '${CALIBRATION_DIR}/public_manifest.json' ]]; then
  python -c 'import hashlib,json,pathlib; from scripts.build_mo_score_calibration_sample import arm_d_fingerprint; root=pathlib.Path(\\\"${CALIBRATION_DIR}\\\"); m=json.load(open(root/\\\"public_manifest.json\\\")); assert m[\\\"audit\\\"][\\\"passed\\\"]; assert arm_d_fingerprint()[\\\"fingerprint\\\"]==m[\\\"arm_d_fingerprint\\\"][\\\"fingerprint\\\"]; expected=m[\\\"secret_artifact_hashes\\\"]; files=(\\\"secret_cases.jsonl\\\",\\\"secret_manifest.jsonl\\\",\\\"engine_snapshot.jsonl\\\",\\\"engine_replay.jsonl\\\"); assert all(hashlib.sha256((root/\\\"secret\\\"/name).read_bytes()).hexdigest()==expected[name] for name in files); print(\\\"FROZEN_SAMPLE_HASH_OK\\\")'
else
  python scripts/build_mo_score_calibration_sample.py \
    --cases ${DATA}/secure_cases/${Y}/${M}/kz_l1_${Y}-${M}-??_cases.jsonl \
    --clinical-csv ${DATA}/secure_cases/${Y}/${M}/mis_protocol_${Y}-${M}.csv \
    --warehouse ${DATA}/warehouse/mo_analytics.sqlite \
    --secret-dir '${CALIBRATION_DIR}/secret' \
    --public-manifest '${CALIBRATION_DIR}/public_manifest.json' \
    --date-from '${FIRST}' --date-to '${LAST}' --target-n 30 --seed 42 --sentinel 3643940
fi
python scripts/eval_mo_score_calibration.py \
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \
  --snapshot '${CALIBRATION_DIR}/secret/engine_snapshot.jsonl' \
  --replay '${CALIBRATION_DIR}/secret/engine_replay.jsonl' \
  --out '${CALIBRATION_DIR}/replay_drift_summary.json'
python scripts/run_mo_calibration_blind_judge.py \
  --cases '${CALIBRATION_DIR}/secret/secret_cases.jsonl' \
  --manifest '${CALIBRATION_DIR}/secret/secret_manifest.jsonl' \
  --out '${JUDGE_OUT}' \
  --summary-out '${SUMMARY_OUT}' \
  ${JUDGE_ARGS}
python -c 'import json; print(json.dumps(json.load(open(\\\"${SUMMARY_OUT}\\\")), ensure_ascii=False))'
\"
echo CALIBRATION_RUN_OK
echo PUBLIC_MANIFEST='${CALIBRATION_DIR}/public_manifest.json'
echo SUMMARY='${SUMMARY_OUT}'
"
  echo "[3/3] done calibration ${MODE} on GCE ${VM}"
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
  -e MO_ICD_LLM_REVIEW='${MO_ICD_LLM_REVIEW:-0}' \
  -e MO_ICD_LLM_REVIEW_LIMIT='${MO_ICD_LLM_REVIEW_LIMIT:-50}' \
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
  -e MO_ICD_LLM_REVIEW='${MO_ICD_LLM_REVIEW:-0}' \
  -e MO_ICD_LLM_REVIEW_LIMIT='${MO_ICD_LLM_REVIEW_LIMIT:-50}' \
  '${CONTAINER}' bash /app/scripts/mo_llm_range_runner.sh
sleep 2
sudo docker exec '${CONTAINER}' pgrep -af 'mo_llm_range_runner|grade_kz_llm' | head -5 || true
echo LOG=${DATA}/logs/mo_llm_backfill_${FIRST}_${LAST}.log
"
fi
echo "[3/3] started on GCE ${VM}"
echo "Tail: gcloud compute ssh ${VM} --zone=${ZONE} --command=\"sudo tail -f ${DATA}/logs/mo_llm_backfill_${FIRST}_${LAST}.log\""
