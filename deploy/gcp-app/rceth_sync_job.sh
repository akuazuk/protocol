#!/usr/bin/env bash
# Rceth ЛС sync on GCE: crawl manifest + download _s.pdf into /var/data/rceth.
# One writer only (flock). Mac is for fixtures/pytest, not bulk.
#
# Manual pilot:
#   RCETH_LIMIT=100 ./deploy/gcp-app/rceth_sync_job.sh
# Full crawl+download+parse (long; resume-safe):
#   RCETH_LIMIT=0 ./deploy/gcp-app/rceth_sync_job.sh
# Resume after crash (skip crawl if manifest fresh enough):
#   RCETH_SKIP_CRAWL=1 RCETH_LIMIT=0 ./deploy/gcp-app/rceth_sync_job.sh
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${RCETH_DATA_ROOT:-/var/data/rceth}"
LOG_DIR="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}/logs"
DAY="$(date -u +%Y-%m-%d)"
CONTAINER="${RCETH_SYNC_CONTAINER:-protocol-web}"
LIMIT="${RCETH_LIMIT:-100}"
THROTTLE="${RCETH_THROTTLE:-0.6}"
INSECURE="${RCETH_INSECURE_SSL:-1}"
MAX_LETTERS="${RCETH_MAX_LETTERS:-}"
HTTP_TIMEOUT="${RCETH_HTTP_TIMEOUT:-30}"
HTTP_RETRIES="${RCETH_HTTP_RETRIES:-3}"
SKIP_CRAWL="${RCETH_SKIP_CRAWL:-0}"
SKIP_DOWNLOAD="${RCETH_SKIP_DOWNLOAD:-0}"
PARSE="${RCETH_PARSE:-1}"
MODE="${RCETH_MODE:-auto}"
mkdir -p "$LOG_DIR" "$DATA/_sync" "$DATA/pdfs/instr" "$DATA/labels"
LOG="${LOG_DIR}/gce-rceth-sync.log"
LOCK="${DATA}/_sync/rceth_sync.lock"
JOB_ENV="${DATA}/_sync/last_job.env"

# Persist knobs for watchdog restarts.
umask 022
cat >"$JOB_ENV" <<EOF
RCETH_MODE=${MODE}
RCETH_LIMIT=${LIMIT}
RCETH_THROTTLE=${THROTTLE}
RCETH_MAX_LETTERS=${MAX_LETTERS}
RCETH_HTTP_TIMEOUT=${HTTP_TIMEOUT}
RCETH_HTTP_RETRIES=${HTTP_RETRIES}
RCETH_INSECURE_SSL=${INSECURE}
RCETH_SYNC_CONTAINER=${CONTAINER}
RCETH_PARSE=${PARSE}
RCETH_SKIP_CRAWL=${SKIP_CRAWL}
RCETH_SKIP_DOWNLOAD=${SKIP_DOWNLOAD}
RCETH_DATA_ROOT=${DATA}
RCETH_PDF_MAX_BYTES=${RCETH_PDF_MAX_BYTES:-8388608}
PROTOCOL_ROOT=${ROOT}
EOF

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "=== rceth_sync ${DAY} SKIP $(date -u +%Y-%m-%dT%H:%M:%SZ) another writer holds ${LOCK} ===" >>"$LOG"
  exit 0
fi

exec >>"$LOG" 2>&1
echo "=== rceth_sync ${DAY} start $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=${MODE} limit=${LIMIT} max_letters=${MAX_LETTERS:-all} skip_crawl=${SKIP_CRAWL} skip_download=${SKIP_DOWNLOAD} parse=${PARSE} ==="

HOST_PY="$(command -v python3)"
APP_ROOT="$ROOT"
SSL_FLAG=()
if [[ "$INSECURE" == "1" || "$INSECURE" == "true" ]]; then
  SSL_FLAG=(--insecure-ssl)
fi

PY_ENV=(
  -e HOME=/tmp
  -e PYTHONPATH=/app
  -e PYTHONUNBUFFERED=1
  -e RCETH_DATA_ROOT="${DATA}"
  -e RCETH_HTTP_TIMEOUT="${HTTP_TIMEOUT}"
  -e RCETH_HTTP_RETRIES="${HTTP_RETRIES}"
  -e RCETH_PDF_MAX_BYTES="${RCETH_PDF_MAX_BYTES:-8388608}"
)

if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  APP_ROOT=/app
  run_py() {
    docker exec \
      -u "$(id -u):$(id -g)" \
      "${PY_ENV[@]}" \
      -w /app \
      "$CONTAINER" python "$@"
  }
else
  PY="${ROOT}/venv-mis/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$HOST_PY"
  fi
  run_py() {
    RCETH_DATA_ROOT="${DATA}" \
      RCETH_HTTP_TIMEOUT="${HTTP_TIMEOUT}" \
      RCETH_HTTP_RETRIES="${HTTP_RETRIES}" \
      RCETH_PDF_MAX_BYTES="${RCETH_PDF_MAX_BYTES:-8388608}" \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" "$@"
  }
fi

cd "$ROOT"

echo "--- preflight ---"
run_py "${APP_ROOT}/scripts/rceth_sync_run.py" --data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" preflight

if [[ "$SKIP_CRAWL" != "1" && "$SKIP_CRAWL" != "true" ]]; then
  echo "--- crawl ---"
  CRAWL_ARGS=(--data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" crawl)
  if [[ -n "${MAX_LETTERS}" ]]; then
    CRAWL_ARGS+=(--max-letters "$MAX_LETTERS")
  fi
  run_py "${APP_ROOT}/scripts/rceth_sync_run.py" "${CRAWL_ARGS[@]}"
else
  echo "--- crawl skipped ---"
fi

if [[ "$SKIP_DOWNLOAD" != "1" && "$SKIP_DOWNLOAD" != "true" ]]; then
  echo "--- download limit=${LIMIT} ---"
  DL_ARGS=(--data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" download)
  if [[ "$LIMIT" != "0" ]]; then
    DL_ARGS+=(--limit "$LIMIT")
  fi
  run_py "${APP_ROOT}/scripts/rceth_sync_run.py" "${DL_ARGS[@]}"
else
  echo "--- download skipped ---"
fi

if [[ "$PARSE" == "1" || "$PARSE" == "true" ]]; then
  echo "--- parse limit=${LIMIT} ---"
  PARSE_ARGS=(--data-root "$DATA" parse)
  if [[ "$LIMIT" != "0" ]]; then
    PARSE_ARGS+=(--limit "$LIMIT")
  fi
  run_py "${APP_ROOT}/scripts/rceth_sync_run.py" "${PARSE_ARGS[@]}"
fi

echo "--- status ---"
run_py "${APP_ROOT}/scripts/rceth_sync_run.py" --data-root "$DATA" status

echo "=== rceth_sync ${DAY} done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
