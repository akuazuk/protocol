#!/usr/bin/env bash
# GCE host: обработать очередь ingest-visit (MIS fetch на хосте, score в protocol-web).
set -euo pipefail
ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
LOAD_MIS="${ROOT}/deploy/gcp-app/load_mis_env.sh"
VENV="${MIS_VENV:-/opt/protocol/venv-mis}"
cd "$ROOT"
if [[ -f "$LOAD_MIS" ]]; then
  # shellcheck disable=SC1090
  source "$LOAD_MIS"
fi
export PYTHONPATH="$ROOT"
export MO_DATA_ROOT="$DATA"
export MO_INGEST_DOCKER_SCORE="${MO_INGEST_DOCKER_SCORE:-1}"
MODE="${1:---once}"
mkdir -p "$DATA/logs"
if [[ "$MODE" == "--loop" ]]; then
  exec "${VENV}/bin/python" "$ROOT/scripts/run_mo_ingest_queue.py" --loop
fi
exec "${VENV}/bin/python" "$ROOT/scripts/run_mo_ingest_queue.py" --once
