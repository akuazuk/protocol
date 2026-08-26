#!/usr/bin/env bash
# After night retry window: alert if yesterday still failed / missing.
# Cron suggestion: 15 3 * * * (UTC), 15 min after retry.
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
ENV_WEB="${ENV_WEB_REMOTE:-/opt/protocol/.env.gcp-staging}"
STATE_DIR="${DATA}/state"
LOG_DIR="${DATA}/logs"
DAY="${MO_NIGHT_DAY:-}"
if [[ -z "$DAY" ]]; then
  DAY="$(python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo("Europe/Minsk"))
print((now.date() - timedelta(days=1)).isoformat())
PY
)"
fi
STATUS_FILE="${STATE_DIR}/gce_night_${DAY}.json"
LAB_STATUS_FILE="${STATE_DIR}/gce_lab_${DAY}.json"
mkdir -p "$LOG_DIR"
exec >>"${LOG_DIR}/gce-night-check.log" 2>&1
echo "======== CHECK day=${DAY} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"

# Self-heal: deploy may leave env 600 as another SSH user.
if [[ -f "$ENV_WEB" ]] && [[ ! -r "$ENV_WEB" ]] && command -v sudo >/dev/null 2>&1; then
  sudo chown "$(whoami):$(whoami)" "$ENV_WEB" 2>/dev/null || true
  sudo chmod 600 "$ENV_WEB" 2>/dev/null || true
fi

load_telegram_env() {
  local key value line
  [[ -f "$ENV_WEB" ]] || return 0
  [[ -r "$ENV_WEB" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|TELEGRAM_NOTIFY_ENABLED|TELEGRAM_INSECURE_SSL)
        if [[ -z "${!key:-}" ]]; then
          value="${value%\"}"; value="${value#\"}"
          value="${value%\'}"; value="${value#\'}"
          export "$key=$value"
        fi
        ;;
    esac
  done < "$ENV_WEB"
}

if [[ ! -f "$STATUS_FILE" ]] || [[ ! -f "$LAB_STATUS_FILE" ]]; then
  detail="missing_status_file"
  if [[ -f "$STATUS_FILE" ]]; then
    detail="missing_lab_status_file"
  fi
  echo "ALERT_NEEDED day=${DAY} detail=${detail}"
  load_telegram_env
  python3 "$ROOT/scripts/telegram_notify.py" \
    "МО GCE night FAIL day=${DAY} mode=check detail=${detail} host=gce" >/dev/null 2>&1 || true
  exit 1
fi

set +e
python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${STATUS_FILE}")
d = json.loads(p.read_text(encoding="utf-8"))
lab_path = Path("${LAB_STATUS_FILE}")
lab = json.loads(lab_path.read_text(encoding="utf-8"))
status = d.get("status") or ""
detail = d.get("detail") or ""
lab_status = lab.get("status") or ""
lab_detail = lab.get("detail") or ""
print(
    f"status={status} detail={detail} lab_status={lab_status} "
    f"lab_detail={lab_detail} sha={(d.get('inbound_sha256') or '')[:12]}"
)
if status == "success" and lab_status == "success":
    print("CHECK_OK")
    sys.exit(0)
print(
    f"ALERT_NEEDED day=${DAY} "
    f"detail=case_{status}_{detail}_lab_{lab_status}_{lab_detail}"
)
sys.exit(2)
PY
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  exit 0
fi
load_telegram_env
detail="$(python3 -c "import json; c=json.load(open('${STATUS_FILE}')); l=json.load(open('${LAB_STATUS_FILE}')); print('case_'+str(c.get('status') or 'missing')+'_lab_'+str(l.get('status') or 'missing'))")"
python3 "$ROOT/scripts/telegram_notify.py" \
  "МО GCE night FAIL day=${DAY} mode=check detail=${detail} host=gce" >/dev/null 2>&1 || true
exit "$rc"
