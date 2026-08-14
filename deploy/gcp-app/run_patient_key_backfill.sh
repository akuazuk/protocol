#!/usr/bin/env bash
# На VM protocol-app: identity backfill patient_key из MIS + refresh history_*.
# Не печатает patient_id. Не гонять с Mac.
set -euo pipefail
ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
# shellcheck disable=SC1091
source "${ROOT}/deploy/gcp-app/load_mis_env.sh"
export PYTHONPATH="${ROOT}"
export MO_WAREHOUSE="${MO_WAREHOUSE:-/var/data/medical_exams/warehouse/mo_analytics.sqlite}"
PY="${ROOT}/venv-mis/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
exec "$PY" "${ROOT}/scripts/backfill_mo_patient_keys_from_mis.py" "$@"
