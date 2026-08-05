#!/usr/bin/env bash
# Прогон night LLM + action-judge на Render (нет geo-block Gemini).
# Пример:
#   bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_HOST="${RENDER_SSH_HOST:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
SSH_ID="${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
FIRST="${1:?first date YYYY-MM-DD}"
LAST="${2:-$FIRST}"

scp -o BatchMode=yes -i "$SSH_ID" \
  "$ROOT/scripts/grade_kz_llm.py" \
  "$ROOT/scripts/run_mo_action_queue_llm_judge.py" \
  "$SSH_HOST:/opt/render/project/src/scripts/"

ssh -o BatchMode=yes -o ServerAliveInterval=60 -i "$SSH_ID" "$SSH_HOST" \
  "FIRST='$FIRST' LAST='$LAST' bash -s" <<'EOS'
set -euo pipefail
cd /opt/render/project/src
DATA=/var/data/medical_exams
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
for d in "${days[@]}"; do
  y=${d:0:4}; m=${d:5:2}; day=${d:8:2}
  echo "=== night grade $d ==="
  .venv/bin/python scripts/grade_kz_llm.py \
    --cases "$DATA/secure_cases/$y/$m/kz_l1_${d}_cases.jsonl" \
    --queue "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_queue.json" \
    --out "$DATA/secure_cases/$y/$m/kz_l1_${d}_llm_grades.jsonl" \
    --warehouse "$DATA/warehouse/mo_analytics.sqlite" \
    --run-id "render-backfill-$d" \
    --escalate --resume --retry-errors
  echo "=== action judge $d ==="
  mkdir -p "$DATA/llm_action_judge/$y/$m/$day"
  .venv/bin/python scripts/run_mo_action_queue_llm_judge.py \
    --date "$d" --source render --stages ab --concurrency 2 --limit 20 \
    --medical-exams-root "$DATA" \
    --out "$DATA/llm_action_judge/$y/$m/$day/judges.jsonl" || true
done
echo ALL_DONE
EOS
