#!/usr/bin/env bash
# Daily КП sync on GCE: crawl + diff. Chunk rebuild only for changed_paths.
# Separate from MIS night (02:00 UTC). Do not load MIS DSN here.
#
# Cron (UTC): 0 1 * * * /opt/protocol/deploy/gcp-app/night_kp_sync.sh
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
CORPUS="${PROTOCOL_CORPUS_ROOT:-/var/data/protocol_corpus}"
LOG_DIR="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}/logs"
SYNC_DIR="${CORPUS}/_sync"
DAY="$(date -u +%Y-%m-%d)"
mkdir -p "$LOG_DIR" "$SYNC_DIR"
LOG="${LOG_DIR}/gce-kp-sync.log"

exec >>"$LOG" 2>&1
echo "=== kp_sync ${DAY} start $(date -u +%Y-%m-%dT%H:%M:%SZ) user=$(whoami) ==="

PY="${ROOT}/venv-mis/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

STAMP="${SYNC_DIR}/kp_sync_${DAY}.ok"
if [[ -f "$STAMP" && "${KP_SYNC_FORCE:-0}" != "1" ]]; then
  echo "ALREADY_OK ${DAY}"
  exit 0
fi

seed_if_missing() {
  local src="$1" dest="$2"
  if [[ ! -f "$dest" && -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -f "$src" "$dest"
  fi
}
seed_if_missing "${ROOT}/data/catalog/protocol_icd_profiles.jsonl" "${CORPUS}/protocol_icd_profiles.jsonl"
seed_if_missing "${ROOT}/data/protocol_catalog.jsonl" "${CORPUS}/protocol_catalog.jsonl"
seed_if_missing "${ROOT}/output/registry/protocol_cards.jsonl" "${CORPUS}/output/registry/protocol_cards.jsonl"

SITE_JSON="${SYNC_DIR}/site_${DAY}.json"
"$PY" "${ROOT}/scripts/kp_sync_run.py" crawl --out "$SITE_JSON"
LOCAL_MANIFEST="${CORPUS}/minzdrav_protocols/_manifest.jsonl"
if [[ ! -f "$LOCAL_MANIFEST" ]]; then
  echo "[]" > "${SYNC_DIR}/empty_manifest.json"
  LOCAL_MANIFEST="${SYNC_DIR}/empty_manifest.json"
fi
OUT="${SYNC_DIR}/kp_sync_${DAY}.json"

"$PY" "${ROOT}/scripts/kp_sync_run.py" diff \
  --site "$SITE_JSON" \
  --local "$LOCAL_MANIFEST" \
  --out "$OUT"

CHANGED="$("$PY" -c "import json; d=json.load(open('${OUT}')); print(len(d.get('changed_paths') or []))")"
echo "changed_paths=${CHANGED}"
if [[ "$CHANGED" == "0" ]]; then
  echo "NO_CHANGES"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$STAMP"
  exit 0
fi

"$PY" "${ROOT}/scripts/kp_sync_run.py" apply \
  --diff "$OUT" \
  --dest "${CORPUS}/minzdrav_protocols" \
  --max-downloads "${KP_SYNC_MAX_DOWNLOADS:-8}"

PATHS_FILE="${SYNC_DIR}/changed_paths_${DAY}.txt"
"$PY" -c "
import json
from pathlib import Path
d=json.load(open('${OUT}'))
Path('${PATHS_FILE}').write_text('\\n'.join(d.get('changed_paths') or [])+'\\n', encoding='utf-8')
"

export CORPUS_PDF_ROOT="${CORPUS}/minzdrav_protocols"
export CORPUS_OUTPUT_ROOT="${CORPUS}/output"
"$PY" -m corpus_pipeline.run_pipeline --changed-only --only-paths "$PATHS_FILE"

CHUNKS="${CORPUS_OUTPUT_ROOT}/chunks/chunks.jsonl"
if [[ -f "$CHUNKS" ]]; then
  "$PY" "${ROOT}/scripts/kp_sync_run.py" merge-indexes \
    --paths "$PATHS_FILE" \
    --chunks "$CHUNKS" \
    --icd-index "${CORPUS}/protocol_icd_profiles.jsonl" \
    --catalog "${CORPUS}/protocol_catalog.jsonl" || echo "WARN: merge-indexes failed"
fi

publish_index() {
  local src="$1" dest="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -f "$src" "$dest" || true
  fi
}
publish_index "${CORPUS}/protocol_catalog.jsonl" "${ROOT}/data/protocol_catalog.jsonl"
publish_index "${CORPUS}/protocol_icd_profiles.jsonl" "${ROOT}/data/catalog/protocol_icd_profiles.jsonl"
publish_index "${CORPUS}/output/registry/protocol_cards.jsonl" "${ROOT}/output/registry/protocol_cards.jsonl"

date -u +%Y-%m-%dT%H:%M:%SZ >"$STAMP"
echo "KP_SYNC_OK changed=${CHANGED}"
