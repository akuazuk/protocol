#!/usr/bin/env bash
# Прогон night LLM + action-judge на Render (нет geo-block Gemini).
# Запускает durable nohup на Render, чтобы SSH disconnect не убивал прогон.
# Пример:
#   bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
#   bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04 --foreground
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_HOST="${RENDER_SSH_HOST:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
SSH_ID="${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
FIRST="${1:?first date YYYY-MM-DD}"
LAST="${2:-$FIRST}"
MODE="${3:-}"

scp -o BatchMode=yes -i "$SSH_ID" \
  "$ROOT/scripts/grade_kz_llm.py" \
  "$ROOT/scripts/run_mo_action_queue_llm_judge.py" \
  "$ROOT/scripts/recompute_mo_days.py" \
  "$SSH_HOST:/opt/render/project/src/scripts/"

remote_body=$(cat <<'EOS'
set +e
cd /opt/render/project/src
DATA=/var/data/medical_exams
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
echo "SUPERVISOR $(date -u) FIRST=$FIRST LAST=$LAST" | tee -a "$LOG"
for d in "${days[@]}"; do
  y=${d:0:4}; m=${d:5:2}; day=${d:8:2}
  echo "=== night grade $d $(date -u) ===" | tee -a "$LOG"
  .venv/bin/python scripts/grade_kz_llm.py \
    --cases "$DATA/secure_cases/$y/$m/kz_l1_${d}_cases.jsonl" \
    --queue "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_queue.json" \
    --out "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_grades.jsonl" \
    --warehouse "$DATA/warehouse/mo_analytics.sqlite" \
    --run-id "render-backfill-$d" \
    --escalate --resume --retry-errors >>"$LOG" 2>&1
  echo "grade_exit_$d=$?" | tee -a "$LOG"
  mkdir -p "$DATA/llm_action_judge/$y/$m/$day"
  .venv/bin/python scripts/run_mo_action_queue_llm_judge.py \
    --date "$d" --source render --stages ab --concurrency 2 --limit 20 \
    --medical-exams-root "$DATA" \
    --out "$DATA/llm_action_judge/$y/$m/$day/judges.jsonl" >>"$LOG" 2>&1
  echo "judge_exit_$d=$?" | tee -a "$LOG"
done
.venv/bin/python scripts/recompute_mo_days.py \
  --data-root "$DATA" \
  --first-date "$FIRST" \
  --last-date "$LAST" \
  --warehouse "$DATA/warehouse/mo_analytics.sqlite" >>"$LOG" 2>&1
echo "ALL_DONE $(date -u)" | tee -a "$LOG"
EOS
)

if [[ "$MODE" == "--foreground" ]]; then
  ssh -o BatchMode=yes -o ServerAliveInterval=60 -i "$SSH_ID" "$SSH_HOST" \
    "FIRST='$FIRST' LAST='$LAST' bash -s" <<<"$remote_body"
else
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -i "$SSH_ID" "$SSH_HOST" \
    "FIRST='$FIRST' LAST='$LAST' bash -s" <<EOS
mkdir -p /var/data/medical_exams/logs
cat > /var/data/medical_exams/logs/run_llm_range.sh <<INNER
$remote_body
INNER
chmod +x /var/data/medical_exams/logs/run_llm_range.sh
# не дублировать, если уже крутится grade для этого диапазона
if pgrep -f "scripts/grade_kz_llm.py" >/dev/null; then
  echo "ALREADY_RUNNING grade_kz_llm"
  pgrep -af "scripts/grade_kz_llm.py" | head -3
  exit 0
fi
nohup env FIRST='$FIRST' LAST='$LAST' bash /var/data/medical_exams/logs/run_llm_range.sh \
  >/var/data/medical_exams/logs/mo_llm_backfill_${FIRST}_${LAST}.nohup 2>&1 &
echo STARTED_PID=\$!
echo LOG=/var/data/medical_exams/logs/mo_llm_backfill_${FIRST}_${LAST}.log
sleep 2
pgrep -af "grade_kz_llm.py|run_llm_range" | head -5 || true
EOS
fi
