#!/usr/bin/env bash
# Durable night LLM + action-judge range runner (Render or GCE).
# Env:
#   FIRST, LAST (YYYY-MM-DD) required
#   SRC_ROOT  default: /opt/render/project/src (Render) or /opt/protocol (GCE)
#   DATA      default: /var/data/medical_exams
#   PYTHON    default: .venv/bin/python if exists else python3
set +e
DATA="${DATA:-/var/data/medical_exams}"
if [[ -z "${SRC_ROOT:-}" ]]; then
  if [[ -d /opt/protocol/scripts ]]; then
    SRC_ROOT=/opt/protocol
  else
    SRC_ROOT=/opt/render/project/src
  fi
fi
cd "$SRC_ROOT" || exit 1
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SRC_ROOT/.venv/bin/python" ]]; then
    PYTHON="$SRC_ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi
FIRST="${FIRST:?FIRST date required}"
LAST="${LAST:?LAST date required}"
mkdir -p "$DATA/logs"
LOG="$DATA/logs/mo_llm_backfill_${FIRST}_${LAST}.log"
mapfile -t days < <(FIRST="$FIRST" LAST="$LAST" python3 - <<'PY'
from datetime import date, timedelta
import os
first = date.fromisoformat(os.environ["FIRST"])
last = date.fromisoformat(os.environ["LAST"])
day = first
while day <= last:
    print(day.isoformat())
    day += timedelta(days=1)
PY
)
echo "SUPERVISOR $(date -u) HOST=${RUN_HOST:-unknown} SRC_ROOT=$SRC_ROOT FIRST=$FIRST LAST=$LAST" | tee -a "$LOG"
for d in "${days[@]}"; do
  y=${d:0:4}; m=${d:5:2}; day=${d:8:2}
  echo "=== night grade $d $(date -u) ===" | tee -a "$LOG"
  "$PYTHON" scripts/grade_kz_llm.py \
    --cases "$DATA/secure_cases/$y/$m/kz_l1_${d}_cases.jsonl" \
    --queue "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_queue.json" \
    --out "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_grades.jsonl" \
    --warehouse "$DATA/warehouse/mo_analytics.sqlite" \
    --run-id "${RUN_ID_PREFIX:-gcp-llm}-$d" \
    --escalate --resume --retry-errors >>"$LOG" 2>&1
  echo "grade_exit_$d=$?" | tee -a "$LOG"
  mkdir -p "$DATA/llm_action_judge/$y/$m/$day"
  JUDGE_LIMIT="${MO_ACTION_JUDGE_LIMIT:-0}"
  "$PYTHON" scripts/run_mo_action_queue_llm_judge.py \
    --date "$d" --source local --stages ab --concurrency 3 --limit "$JUDGE_LIMIT" \
    --medical-exams-root "$DATA" \
    --out "$DATA/llm_action_judge/$y/$m/$day/judges.jsonl" >>"$LOG" 2>&1
  echo "judge_exit_$d=$?" | tee -a "$LOG"
  # Shadow Dx/Plan (option B): conservative attention; does not touch SSOT
  if [[ "${MO_SHADOW_DX_PLAN:-1}" == "1" || "${MO_SHADOW_DX_PLAN:-1}" == "true" ]]; then
    mkdir -p "$DATA/llm_shadow_dx_plan/$y/$m/$day"
    SHADOW_LIMIT="${MO_SHADOW_DX_PLAN_LIMIT:-0}"
    echo "=== shadow dx/plan $d limit=$SHADOW_LIMIT $(date -u) ===" | tee -a "$LOG"
    "$PYTHON" scripts/run_mo_shadow_dx_plan.py \
      --date "$d" --medical-exams-root "$DATA" --resume --concurrency 2 \
      --limit "$SHADOW_LIMIT" \
      --warehouse "$DATA/warehouse/mo_analytics.sqlite" \
      --out "$DATA/llm_shadow_dx_plan/$y/$m/$day/shadow.jsonl" >>"$LOG" 2>&1
    echo "shadow_dx_plan_exit_$d=$?" | tee -a "$LOG"
  else
    echo "shadow_dx_plan_skip_$d=flag_off" | tee -a "$LOG"
  fi
  # ICD grey-zone LLM review (фаза 4): default off
  if [[ "${MO_ICD_LLM_REVIEW:-0}" == "1" || "${MO_ICD_LLM_REVIEW:-0}" == "true" ]]; then
    mkdir -p "$DATA/llm_icd_review/$y/$m/$day"
    ICD_LIMIT="${MO_ICD_LLM_REVIEW_LIMIT:-50}"
    echo "=== icd llm review $d limit=$ICD_LIMIT $(date -u) ===" | tee -a "$LOG"
    MO_ICD_LLM_REVIEW=1 "$PYTHON" scripts/run_mo_icd_llm_review.py \
      --date "$d" --medical-exams-root "$DATA" --limit "$ICD_LIMIT" --force-enable \
      --out "$DATA/llm_icd_review/$y/$m/$day/reviews.jsonl" >>"$LOG" 2>&1
    echo "icd_llm_review_exit_$d=$?" | tee -a "$LOG"
  else
    echo "icd_llm_review_skip_$d=flag_off" | tee -a "$LOG"
  fi
done
"$PYTHON" scripts/recompute_mo_days.py \
  --data-root "$DATA" \
  --first-date "$FIRST" \
  --last-date "$LAST" \
  --warehouse "$DATA/warehouse/mo_analytics.sqlite" >>"$LOG" 2>&1
echo "ALL_DONE $(date -u)" | tee -a "$LOG"
