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
  # Container worker (real parse/crawl/download). Avoid matching this shell's argv.
  if docker top "$CONTAINER" 2>/dev/null | grep -E 'rceth_sync_run\.py' >/dev/null 2>&1; then
    return 0
  fi
  # Host wrapper: only bash executing the job script, not SSH helpers that mention the path.
  if pgrep -f 'bash (/opt/protocol/deploy/gcp-app|/var/data/rceth/_code/scripts)/rceth_sync_job\.sh' >/dev/null 2>&1; then
    return 0
  fi
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
    # Process already confirmed dead above. Do not wait for heartbeat stale -
    # otherwise UI stays interrupted for minutes after container kill/OOM.
    if [[ -n "$AGE" && "$AGE" != "None" && "$AGE" -lt 45 ]]; then
      log "wait: status=${STATUS_NAME} just updated (age=${AGE}s), grace before restart"
      exit 0
    fi
    NEED=1
    REASON="running_without_process age=${AGE}"
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
  # Синтез knobs из status + volume (пилот мог стартовать без last_job.env).
  python3 - "$STATUS" "$DATA" "$JOB_ENV" <<'PY'
import json, os, sys
from pathlib import Path
status_path, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
total, phase = 0, ""
if status_path.is_file():
    try:
        d = json.loads(status_path.read_text(encoding="utf-8"))
        prog = d.get("progress") if isinstance(d.get("progress"), dict) else {}
        total = int(prog.get("total") or 0)
        phase = str(d.get("phase") or "")
    except Exception:
        pass
pdf_n = len(list((data / "pdfs" / "instr").glob("*_s.pdf"))) if (data / "pdfs" / "instr").is_dir() else 0
labels_n = len(list((data / "labels").glob("*.json"))) if (data / "labels").is_dir() else 0
manifest = data / "manifest.jsonl"
has_manifest = manifest.is_file() and manifest.stat().st_size > 0
if not has_manifest and pdf_n == 0:
    raise SystemExit("no_volume")
# Prefer parse-only resume when PDFs already on disk.
limit = total if total > 0 else (50 if pdf_n and pdf_n <= 200 else 0)
skip_dl = "1" if pdf_n > 0 else "0"
skip_crawl = "1" if has_manifest else "0"
lines = [
    "RCETH_MODE=resume",
    f"RCETH_LIMIT={limit}",
    "RCETH_THROTTLE=0.6",
    "RCETH_MAX_LETTERS=",
    "RCETH_HTTP_TIMEOUT=30",
    "RCETH_HTTP_RETRIES=3",
    "RCETH_INSECURE_SSL=1",
    "RCETH_SYNC_CONTAINER=protocol-web",
    "RCETH_PARSE=1",
    f"RCETH_SKIP_CRAWL={skip_crawl}",
    f"RCETH_SKIP_DOWNLOAD={skip_dl}",
    f"RCETH_DATA_ROOT={data}",
    "RCETH_PDF_MAX_BYTES=8388608",
    "PROTOCOL_ROOT=/opt/protocol",
]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"synthesized limit={limit} pdfs={pdf_n} labels={labels_n} phase={phase} skip_dl={skip_dl}")
PY
  syn_rc=$?
  if [[ "$syn_rc" -ne 0 ]]; then
    log "skip: need restart (${REASON}) but cannot synthesize ${JOB_ENV}"
    exit 0
  fi
  log "synthesized ${JOB_ENV}"
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

# Resume: skip crawl by default; download resume-safe; parse skips existing labels.
# shellcheck disable=SC1090
set -a
RCETH_SKIP_CRAWL=1
RCETH_SKIP_DOWNLOAD=0
RCETH_PARSE=1
# shellcheck source=/dev/null
source "$JOB_ENV"
set +a
export RCETH_SKIP_CRAWL="${RCETH_SKIP_CRAWL:-1}"
export RCETH_DATA_ROOT="${RCETH_DATA_ROOT:-$DATA}"
export RCETH_SYNC_CONTAINER="${RCETH_SYNC_CONTAINER:-$CONTAINER}"
# If PDFs already present, skip re-download on crash recovery.
if [[ -d "${RCETH_DATA_ROOT}/pdfs/instr" ]] && ls "${RCETH_DATA_ROOT}/pdfs/instr"/*_s.pdf >/dev/null 2>&1; then
  export RCETH_SKIP_DOWNLOAD=1
fi

log "restart #${COUNT}: ${REASON} script=${JOB_SCRIPT} limit=${RCETH_LIMIT:-?} mode=${RCETH_MODE:-?} skip_dl=${RCETH_SKIP_DOWNLOAD}"
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
  RCETH_SKIP_CRAWL="${RCETH_SKIP_CRAWL:-1}" \
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
