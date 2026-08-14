#!/usr/bin/env bash
# Watchdog rceth sync на GCE: если job умер, а status=running/interrupted - resume.
# Один writer (тот же flock в rceth_sync_job.sh). Не стартует новый full без last_job.env.
#
# On VM (cron every 10 min):
#   bash /opt/protocol/deploy/gcp-app/rceth_sync_watchdog.sh
# Dry-run:
#   RCETH_WATCHDOG_DRY=1 bash deploy/gcp-app/rceth_sync_watchdog.sh
set -euo pipefail

DATA="${RCETH_DATA_ROOT:-/var/data/rceth}"
LOG_DIR="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}/logs"
CONTAINER="${RCETH_SYNC_CONTAINER:-protocol-web}"
STATUS="${DATA}/_sync/status.json"
JOB_ENV="${DATA}/_sync/last_job.env"
STATE="${DATA}/_sync/watchdog_state.json"
LOG="${LOG_DIR}/gce-rceth-watchdog.log"
JOB_SCRIPT="${RCETH_JOB_SCRIPT:-/opt/protocol/deploy/gcp-app/rceth_sync_job.sh}"
if [[ ! -x "$JOB_SCRIPT" && -x /var/data/rceth/_code/scripts/rceth_sync_job.sh ]]; then
  JOB_SCRIPT=/var/data/rceth/_code/scripts/rceth_sync_job.sh
fi
STALE_SEC="${RCETH_STATUS_STALE_SEC:-300}"
MAX_RESTARTS_DAY="${RCETH_WATCHDOG_MAX_RESTARTS_DAY:-6}"
DRY="${RCETH_WATCHDOG_DRY:-0}"
mkdir -p "$LOG_DIR" "$DATA/_sync"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$LOG" >&2; }

process_alive() {
  docker top "$CONTAINER" 2>/dev/null | grep -q rceth_sync_run && return 0
  # host-side nohup bash job (download/crawl wrapper)
  pgrep -f 'rceth_sync_job.sh' >/dev/null 2>&1 && return 0
  return 1
}

read_status_json() {
  python3 - "$STATUS" "$STALE_SEC" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, stale_sec = sys.argv[1], int(sys.argv[2])
out = {"exists": False, "status": "", "phase": "", "age_sec": None, "stale": False, "done": 0, "total": 0}
if not os.path.isfile(path):
    print(json.dumps(out)); raise SystemExit(0)
try:
    d = json.load(open(path, encoding="utf-8"))
except Exception:
    print(json.dumps(out)); raise SystemExit(0)
out["exists"] = True
out["status"] = str(d.get("status") or "")
out["phase"] = str(d.get("phase") or "")
prog = d.get("progress") if isinstance(d.get("progress"), dict) else {}
out["done"] = int(prog.get("done") or 0)
out["total"] = int(prog.get("total") or 0)
raw = str(d.get("updated_at") or "")
if raw.endswith("Z"):
    raw = raw[:-1] + "+00:00"
try:
    updated = datetime.fromisoformat(raw)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age = int((datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds())
    out["age_sec"] = age
    out["stale"] = age > stale_sec
except Exception:
    out["stale"] = True
print(json.dumps(out))
PY
}

bump_restarts() {
  python3 - "$STATE" "$MAX_RESTARTS_DAY" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, max_n = sys.argv[1], int(sys.argv[2])
day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
data = {"day": day, "restarts": 0}
if os.path.isfile(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        pass
if data.get("day") != day:
    data = {"day": day, "restarts": 0}
n = int(data.get("restarts") or 0)
if n >= max_n:
    print("blocked"); raise SystemExit(0)
data["restarts"] = n + 1
data["last_restart_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
open(path, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(str(data["restarts"]))
PY
}

if process_alive; then
  log "ok: process alive"
  exit 0
fi

INFO="$(read_status_json)"
STATUS_NAME="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status") or "")' "$INFO")"
STALE="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1]).get("stale") else "0")' "$INFO")"
AGE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("age_sec"))' "$INFO")"
PHASE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("phase") or "")' "$INFO")"

NEED=0
REASON=""
case "$STATUS_NAME" in
  running|queued)
    if [[ "$STALE" == "1" ]]; then
      NEED=1
      REASON="running_stale age=${AGE}"
    else
      log "wait: status=${STATUS_NAME} but no process yet (age=${AGE})"
      exit 0
    fi
    ;;
  interrupted|error)
    NEED=1
    REASON="status=${STATUS_NAME} phase=${PHASE}"
    ;;
  done|idle|"")
    log "ok: status=${STATUS_NAME:-missing} no restart"
    exit 0
    ;;
  *)
    log "ok: status=${STATUS_NAME} no restart"
    exit 0
    ;;
esac

if [[ ! -f "$JOB_ENV" ]]; then
  log "skip: need restart (${REASON}) but missing ${JOB_ENV}"
  exit 0
fi
if [[ ! -x "$JOB_SCRIPT" ]]; then
  log "skip: job script missing ${JOB_SCRIPT}"
  exit 1
fi

COUNT="$(bump_restarts)"
if [[ "$COUNT" == "blocked" ]]; then
  log "blocked: max restarts/day (${MAX_RESTARTS_DAY}) reached"
  exit 0
fi

# Resume: skip crawl (manifest already on disk); download is resume-safe.
# shellcheck disable=SC1090
set -a
# defaults then overlay last job
RCETH_SKIP_CRAWL=1
RCETH_SKIP_DOWNLOAD=0
RCETH_PARSE=1
# shellcheck source=/dev/null
source "$JOB_ENV"
set +a
# Force resume knobs for crash recovery (keep LIMIT/MODE from last job).
export RCETH_SKIP_CRAWL=1
export RCETH_DATA_ROOT="${RCETH_DATA_ROOT:-$DATA}"
export RCETH_SYNC_CONTAINER="${RCETH_SYNC_CONTAINER:-$CONTAINER}"

log "restart #${COUNT}: ${REASON} script=${JOB_SCRIPT} limit=${RCETH_LIMIT:-?} mode=${RCETH_MODE:-?}"
if [[ "$DRY" == "1" || "$DRY" == "true" ]]; then
  log "dry-run: not starting"
  exit 0
fi

nohup env \
  RCETH_MODE="${RCETH_MODE:-resume}" \
  RCETH_LIMIT="${RCETH_LIMIT:-0}" \
  RCETH_THROTTLE="${RCETH_THROTTLE:-0.6}" \
  RCETH_MAX_LETTERS="${RCETH_MAX_LETTERS:-}" \
  RCETH_HTTP_TIMEOUT="${RCETH_HTTP_TIMEOUT:-30}" \
  RCETH_HTTP_RETRIES="${RCETH_HTTP_RETRIES:-3}" \
  RCETH_INSECURE_SSL="${RCETH_INSECURE_SSL:-1}" \
  RCETH_SYNC_CONTAINER="${RCETH_SYNC_CONTAINER}" \
  RCETH_PARSE="${RCETH_PARSE:-1}" \
  RCETH_SKIP_CRAWL=1 \
  RCETH_SKIP_DOWNLOAD="${RCETH_SKIP_DOWNLOAD:-0}" \
  RCETH_DATA_ROOT="${RCETH_DATA_ROOT}" \
  RCETH_PDF_MAX_BYTES="${RCETH_PDF_MAX_BYTES:-8388608}" \
  PROTOCOL_ROOT="${PROTOCOL_ROOT:-/opt/protocol}" \
  bash "$JOB_SCRIPT" \
  >>"$LOG" 2>&1 &
log "started host_pid=$!"
sleep 2
if process_alive; then
  log "ok: process up after restart"
else
  log "warn: process not visible yet (check gce-rceth-sync.log)"
fi
