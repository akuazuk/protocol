#!/usr/bin/env bash
# Прогон night LLM + action-judge на Render (legacy, пока Render = warehouse leader).
# Предпочтительный путь E1 staging: deploy/gcp-llm/run_on_gce.sh
# Не запускать grade_kz_llm локально на Mac. Перед вызовом: vanya_vpn ensure-off.
# Пример:
#   bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04
#   bash scripts/run_mo_render_llm_backfill.sh 2026-08-01 2026-08-04 --foreground
#   bash deploy/gcp-llm/run_on_gce.sh 2026-08-06
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$HOME/CURSOR/bin/vanya_vpn.sh" ]]; then
  "$HOME/CURSOR/bin/vanya_vpn.sh" ensure-off >/dev/null 2>&1 || true
fi
SSH_HOST="${RENDER_SSH_HOST:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
SSH_ID="${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
FIRST="${1:?first date YYYY-MM-DD}"
LAST="${2:-$FIRST}"
MODE="${3:-}"

# Статический runner в репо: без nested heredoc (remote <<INNER схлопывал $DATA/$d).
scp -o BatchMode=yes -i "$SSH_ID" \
  "$ROOT/scripts/mo_llm_range_runner.sh" \
  "$ROOT/scripts/grade_kz_llm.py" \
  "$ROOT/scripts/run_mo_action_queue_llm_judge.py" \
  "$ROOT/scripts/recompute_mo_days.py" \
  "$SSH_HOST:/opt/render/project/src/scripts/"

if [[ "$MODE" == "--foreground" ]]; then
  ssh -o BatchMode=yes -o ServerAliveInterval=60 -i "$SSH_ID" "$SSH_HOST" \
    "chmod +x /opt/render/project/src/scripts/mo_llm_range_runner.sh && FIRST='$FIRST' LAST='$LAST' bash /opt/render/project/src/scripts/mo_llm_range_runner.sh"
else
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -i "$SSH_ID" "$SSH_HOST" \
    "chmod +x /opt/render/project/src/scripts/mo_llm_range_runner.sh
mkdir -p /var/data/medical_exams/logs
# Только реальный python grade (не bash -c / pgrep с тем же путём в cmdline).
if pgrep -f '[.]venv/bin/python .*grade_kz_llm[.]py' >/dev/null; then
  echo 'ALREADY_RUNNING grade_kz_llm'
  pgrep -af '[.]venv/bin/python .*grade_kz_llm[.]py' | head -3
  exit 0
fi
nohup env FIRST='$FIRST' LAST='$LAST' bash /opt/render/project/src/scripts/mo_llm_range_runner.sh \
  >/var/data/medical_exams/logs/mo_llm_backfill_${FIRST}_${LAST}.nohup 2>&1 &
echo STARTED_PID=\$!
echo LOG=/var/data/medical_exams/logs/mo_llm_backfill_${FIRST}_${LAST}.log
sleep 3
pgrep -af 'grade_kz_llm[.]py|mo_llm_range_runner' | head -5 || true
"
fi
