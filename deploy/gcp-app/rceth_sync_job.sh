#!/usr/bin/env bash
# Rceth ЛС sync on GCE: crawl manifest + download _s.pdf into /var/data/rceth.
# One writer only. Mac is for fixtures/pytest, not bulk.
#
# Manual:
#   RCETH_LIMIT=100 ./deploy/gcp-app/rceth_sync_job.sh
# Full crawl+download (long):
#   RCETH_LIMIT=0 ./deploy/gcp-app/rceth_sync_job.sh
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
mkdir -p "$LOG_DIR" "$DATA/_sync" "$DATA/pdfs/instr"
LOG="${LOG_DIR}/gce-rceth-sync.log"

exec >>"$LOG" 2>&1
echo "=== rceth_sync ${DAY} start $(date -u +%Y-%m-%dT%H:%M:%SZ) limit=${LIMIT} max_letters=${MAX_LETTERS:-all} timeout=${HTTP_TIMEOUT} retries=${HTTP_RETRIES} ==="

HOST_PY="$(command -v python3)"
APP_ROOT="$ROOT"
SSL_FLAG=()
if [[ "$INSECURE" == "1" || "$INSECURE" == "true" ]]; then
  SSL_FLAG=(--insecure-ssl)
fi

if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  APP_ROOT=/app
  run_py() {
    docker exec \
      -u "$(id -u):$(id -g)" \
      -e HOME=/tmp \
      -e PYTHONPATH=/app \
      -e PYTHONUNBUFFERED=1 \
      -e RCETH_DATA_ROOT="${DATA}" \
      -e RCETH_HTTP_TIMEOUT="${HTTP_TIMEOUT}" \
      -e RCETH_HTTP_RETRIES="${HTTP_RETRIES}" \
      -w /app \
      "$CONTAINER" python "$@"
  }
else
  PY="${ROOT}/venv-mis/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$HOST_PY"
  fi
  run_py() {
    RCETH_DATA_ROOT="${DATA}" PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$@"
  }
fi

cd "$ROOT"

echo "--- preflight ---"
run_py "${APP_ROOT}/scripts/rceth_sync_run.py" --data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" preflight

echo "--- crawl ---"
CRAWL_ARGS=(--data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" crawl)
if [[ -n "${MAX_LETTERS}" ]]; then
  CRAWL_ARGS+=(--max-letters "$MAX_LETTERS")
fi
run_py "${APP_ROOT}/scripts/rceth_sync_run.py" "${CRAWL_ARGS[@]}"

echo "--- download limit=${LIMIT} ---"
DL_ARGS=(--data-root "$DATA" --throttle "$THROTTLE" --timeout "$HTTP_TIMEOUT" --retries "$HTTP_RETRIES" "${SSL_FLAG[@]}" download)
if [[ "$LIMIT" != "0" ]]; then
  DL_ARGS+=(--limit "$LIMIT")
fi
run_py "${APP_ROOT}/scripts/rceth_sync_run.py" "${DL_ARGS[@]}"

if [[ "${RCETH_PARSE:-1}" == "1" || "${RCETH_PARSE:-1}" == "true" ]]; then
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
