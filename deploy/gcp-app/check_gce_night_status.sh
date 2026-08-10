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
mkdir -p "$LOG_DIR"
exec >>"${LOG_DIR}/gce-night-check.log" 2>&1
echo "======== CHECK day=${DAY} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"

load_telegram_env() {
  local key value line
  [[ -f "$ENV_WEB" ]] || return 0
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

if [[ ! -f "$STATUS_FILE" ]]; then
  detail="missing_status_file"
  echo "ALERT_NEEDED day=${DAY} detail=${detail}"
  load_telegram_env
  python3 "$ROOT/scripts/telegram_notify.py" \
    "МО GCE night FAIL day=${DAY} mode=check detail=${detail} host=gce" >/dev/null 2>&1 || true
  exit 1
fi

python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${STATUS_FILE}")
d = json.loads(p.read_text(encoding="utf-8"))
status = d.get("status") or ""
detail = d.get("detail") or ""
print(f"status={status} detail={detail} sha={(d.get('inbound_sha256') or '')[:12]}")
if status == "success":
    print("CHECK_OK")
    sys.exit(0)
print(f"ALERT_NEEDED day=${DAY} detail=status_{status}_{detail}")
sys.exit(2)
PY
rc=$?
if [[ "$rc" -eq 0 ]]; then
  exit 0
fi
load_telegram_env
detail="$(python3 -c "import json; d=json.load(open('${STATUS_FILE}')); print(d.get('detail') or d.get('status') or 'failed')")"
python3 "$ROOT/scripts/telegram_notify.py" \
  "МО GCE night FAIL day=${DAY} mode=check detail=${detail} host=gce" >/dev/null 2>&1 || true
exit "$rc"
