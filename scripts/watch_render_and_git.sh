#!/usr/bin/env bash
# Monitor origin/main (other workstation) and Render /var/data; optional Telegram alert.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.cursor/render-watch-state.json"
ENV_FILE="$ROOT/.env"
SSH_TARGET="${RENDER_SSH_TARGET:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
OTHER_AUTHOR="${WATCH_OTHER_AUTHOR:-akuazuk@gmail.com}"
THIS_AUTHOR="${WATCH_THIS_AUTHOR:-pavel@iMac-Petya.local}"
TODAY="${WATCH_DATE:-$(date +%Y-%m-%d)}"
INTERVAL="${WATCH_INTERVAL_SEC:-300}"
LOOP="${WATCH_LOOP:-0}"

mkdir -p "$(dirname "$STATE")"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

load_env() {
  TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
  TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
}

send_telegram() {
  local text="$1"
  load_env
  if [[ -z "$TELEGRAM_BOT_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
    echo "TELEGRAM_SKIP (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)"
    return 0
  fi
  if curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${text}" \
    --data-urlencode "disable_web_page_preview=true" | grep -q '"ok":true'; then
    echo "TELEGRAM_SENT"
  else
    echo "TELEGRAM_FAIL" >&2
    return 1
  fi
}

check_git() {
  git -C "$ROOT" fetch origin main --quiet 2>/dev/null || true
  git -C "$ROOT" ls-remote origin main 2>/dev/null | awk '{print $1}'
}

check_disk_sig() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
    -i "${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}" "$SSH_TARGET" \
    "find /var/data -type f -newermt '${TODAY} 00:00:00' -printf '%T@ %s %p\n' 2>/dev/null | sort | md5sum | awk '{print \$1}'" \
    2>/dev/null || echo "ssh_error"
}

list_disk_today() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 \
    -i "${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}" "$SSH_TARGET" \
    "find /var/data -type f -newermt '${TODAY} 00:00:00' -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' 2>/dev/null | sort -r | head -25" \
    2>/dev/null || echo "(ssh unavailable)"
}

latest_other_commit() {
  git -C "$ROOT" log origin/main --author="$OTHER_AUTHOR" -1 --format='%H|%ae|%an|%s|%ci' 2>/dev/null || true
}

init_state() {
  if [[ ! -f "$STATE" ]]; then
    REMOTE_SHA="$(check_git)"
    DISK_SIG="$(check_disk_sig)"
    OTHER_LINE="$(latest_other_commit)"
    OTHER_SHA="${OTHER_LINE%%|*}"
    python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

p = Path("$STATE")
p.write_text(json.dumps({
    "remote_sha": "$REMOTE_SHA",
    "last_other_commit_sha": "$OTHER_SHA",
    "disk_sig": "$DISK_SIG",
    "since": datetime.now().astimezone().isoformat(timespec="seconds"),
}, ensure_ascii=False, indent=2) + "\n")
PY
  fi
}

run_once() {
  init_state
  OLD_SHA="$(python3 -c "import json; print(json.load(open('$STATE')).get('remote_sha',''))")"
  OLD_OTHER="$(python3 -c "import json; print(json.load(open('$STATE')).get('last_other_commit_sha',''))")"
  OLD_DISK="$(python3 -c "import json; print(json.load(open('$STATE')).get('disk_sig',''))")"

  NEW_SHA="$(check_git)"
  NEW_DISK="$(check_disk_sig)"
  OTHER_LINE="$(latest_other_commit)"
  OTHER_SHA="${OTHER_LINE%%|*}"

  OTHER_CHANGED=0
  DISK_CHANGED=0
  if [[ -n "$OTHER_SHA" && "$OTHER_SHA" != "$OLD_OTHER" ]]; then
    OTHER_CHANGED=1
  fi
  if [[ -n "$NEW_DISK" && "$NEW_DISK" != "ssh_error" && "$NEW_DISK" != "$OLD_DISK" ]]; then
    DISK_FILES="$(list_disk_today | { grep -v '^$' || true; } | wc -l | tr -d ' ')"
    if [[ "$DISK_FILES" -gt 0 ]]; then
      DISK_CHANGED=1
    elif [[ "$NEW_DISK" != "d41d8cd98f00b204e9800998ecf8427e" ]]; then
      DISK_CHANGED=1
    fi
  fi

  if [[ "$OTHER_CHANGED" -eq 0 && "$DISK_CHANGED" -eq 0 ]]; then
    echo "RENDER_WATCH_OK $(date +%H:%M:%S) other=${OTHER_SHA:0:8} disk=${NEW_DISK:0:8}"
    python3 - <<PY
import json
from pathlib import Path
p = Path("$STATE")
d = json.loads(p.read_text())
d["remote_sha"] = "$NEW_SHA"
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
PY
    return 0
  fi

  MSG=""
  if [[ "$OTHER_CHANGED" -eq 1 ]]; then
    IFS='|' read -r _sha _email _name _subj _when <<< "$OTHER_LINE"
    MSG+="Protocol: коммит с другого ПК
${_when}
${_name} <${_email}>
${_subj}
${_sha:0:12}
"
    echo "OTHER_COMMIT: $OTHER_LINE"
  fi

  if [[ "$DISK_CHANGED" -eq 1 ]]; then
    DISK_LIST="$(list_disk_today | head -8)"
    MSG+="Protocol: запись на Render /var/data (${TODAY})
${DISK_LIST}
"
    echo "DISK_TODAY:"
    echo "$DISK_LIST"
  fi

  if [[ -n "$MSG" ]]; then
    send_telegram "$MSG"
  fi

  python3 - <<PY
import json
from datetime import datetime
from pathlib import Path
p = Path("$STATE")
d = json.loads(p.read_text()) if p.exists() else {}
d["remote_sha"] = "$NEW_SHA"
d["last_other_commit_sha"] = "$OTHER_SHA"
d["disk_sig"] = "$NEW_DISK"
d["last_alert"] = datetime.now().isoformat(timespec="seconds")
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
PY
  echo "RENDER_WATCH_ALERT $(date +%H:%M:%S)"
}

if [[ "$LOOP" == "1" ]]; then
  echo "watch loop every ${INTERVAL}s (other=$OTHER_AUTHOR, telegram=${TELEGRAM_BOT_TOKEN:+on}${TELEGRAM_BOT_TOKEN:-off})"
  while true; do
    run_once || true
    sleep "$INTERVAL"
  done
else
  run_once
fi
