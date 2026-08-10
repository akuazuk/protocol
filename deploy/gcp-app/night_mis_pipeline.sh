#!/usr/bin/env bash
# Night MIS extract + score on GCE (E2). Mac is not used for SQL.
#
# Modes:
#   main   - always run (cron 02:00 server/UTC)
#   retry  - only if yesterday main did not succeed (cron 03:00 = +1h)
#
# On VM:
#   bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh main
#   bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh retry
set -euo pipefail

MODE="${1:-main}"
ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
ENV_MIS="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
VENV="${MIS_VENV:-/opt/protocol/venv-mis}"
LOG_DIR="${DATA}/logs"
STATE_DIR="${DATA}/state"
# Clinic calendar day (Belarus), not UTC date at 02:00.
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
Y="${DAY:0:4}"
M="${DAY:5:2}"
STATUS_FILE="${STATE_DIR}/gce_night_${DAY}.json"
STAGING="${DATA}/staging/gce-night-${DAY}-$$"
INBOUND="${DATA}/inbound/extract"
LOCK="${STATE_DIR}/gce-night.lock"
WITH_LLM="${MO_NIGHT_WITH_LLM:-1}"
DB_ATTEMPTS="${MO_DB_RETRIES:-5}"
DB_DELAY="${MO_DB_RETRY_DELAY_SEC:-5}"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$INBOUND" 2>/dev/null || true
if [[ ! -w "$LOG_DIR" ]] || [[ ! -w "$STATE_DIR" ]]; then
  sudo mkdir -p "$LOG_DIR" "$STATE_DIR" "$INBOUND" "${DATA}/staging"
  sudo chown -R "$(whoami):$(whoami)" "$LOG_DIR" "$STATE_DIR" "$INBOUND" "${DATA}/staging"
fi
exec >>"${LOG_DIR}/gce-night-${MODE}.log" 2>&1
echo "======== NIGHT ${MODE} day=${DAY} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"

if [[ ! -f "$ENV_MIS" ]]; then
  echo "ERROR: missing $ENV_MIS" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing MIS venv at $VENV (run setup_mis_venv.sh)" >&2
  exit 2
fi

clear_stale_lock() {
  local owner=""
  [[ -f "$LOCK" ]] || return 0
  owner="$(tr -d '[:space:]' < "$LOCK" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    return 0
  fi
  rm -f "$LOCK"
}
acquire_lock() {
  clear_stale_lock
  if (set -o noclobber; printf '%s\n' "$$" >"$LOCK") 2>/dev/null; then
    return 0
  fi
  local owner
  owner="$(cat "$LOCK" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    echo "already running pid=$owner; skip $MODE"
    exit 0
  fi
  rm -f "$LOCK"
  (set -o noclobber; printf '%s\n' "$$" >"$LOCK") 2>/dev/null || exit 0
}
release_lock() {
  if [[ "$(cat "$LOCK" 2>/dev/null || true)" = "$$" ]]; then
    rm -f "$LOCK"
  fi
  rm -rf "$STAGING"
}
acquire_lock
trap release_lock EXIT INT TERM

write_status() {
  local status="$1"
  local detail="${2:-}"
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
path = Path("${STATUS_FILE}")
payload = {
  "day": "${DAY}",
  "mode": "${MODE}",
  "status": "${status}",
  "detail": """${detail}""",
  "updated_at": datetime.now(timezone.utc).isoformat(),
  "host": "gce",
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {path} status={payload['status']}")
PY
}

if [[ "$MODE" == "retry" ]]; then
  if [[ -f "$STATUS_FILE" ]]; then
    prev="$(python3 -c "import json; print(json.load(open('${STATUS_FILE}')).get('status',''))")"
    if [[ "$prev" == "success" ]]; then
      echo "retry skip: day=${DAY} already success"
      exit 0
    fi
    echo "retry: previous status=${prev}"
  else
    echo "retry: no status file; will run"
  fi
elif [[ "$MODE" != "main" ]]; then
  echo "Unknown mode: $MODE (main|retry)" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_MIS"
set +a
export PYTHONPATH="$ROOT"
export MO_DATA_ROOT="$DATA"
export RUN_HOST=gcp

mkdir -p "$STAGING"
TAG="${DAY}"
# exporter tag = day_next
NEXT="$(python3 -c "from datetime import date,timedelta; d=date.fromisoformat('${DAY}'); print((d+timedelta(days=1)).isoformat())")"
EXPORT_CSV="${STAGING}/mis_protocol_${DAY}_${NEXT}.csv"
EXPORT_META="${STAGING}/mis_protocol_${DAY}_${NEXT}.meta.json"

export_ok=0
attempt=1
while [[ "$attempt" -le "$DB_ATTEMPTS" ]]; do
  echo "export attempt ${attempt}/${DB_ATTEMPTS}"
  if "${VENV}/bin/python" "$ROOT/scripts/export_mis_protocol_month.py" \
      --from "$DAY" --to "$NEXT" --out-dir "$STAGING"; then
    export_ok=1
    break
  fi
  echo "export failed attempt=${attempt}"
  sleep $(( DB_DELAY * (2 ** (attempt - 1)) ))
  attempt=$((attempt + 1))
done

if [[ "$export_ok" != "1" ]] || [[ ! -f "$EXPORT_CSV" ]]; then
  write_status "failed" "export_failed_after_${DB_ATTEMPTS}"
  exit 1
fi

# doctor join sanity (soft block like Mac gate: <50% on >=20 rows)
"${VENV}/bin/python" - <<PY
import csv, json, sys
from pathlib import Path
csv_path = Path("${EXPORT_CSV}")
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
n = len(rows)
fio = sum(1 for r in rows if (r.get("doctor_fio") or "").strip())
pct = (100.0 * fio / n) if n else 100.0
print(f"doctor_join rows={n} with_fio={fio} pct={pct:.1f}")
meta = {
  "schema_version": 1,
  "day": "${DAY}",
  "extracted_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
  "run_host": "gcp",
  "row_count": n,
  "doctor_fio_pct": round(pct, 2),
  "source": "kravira_mc.mis_protocol + mis_data",
  "mode": "${MODE}",
}
if n >= 20 and pct < 50.0:
  print("ERROR: doctor_join_broken", flush=True)
  Path("${STATUS_FILE}").write_text(json.dumps({
    "day": "${DAY}", "mode": "${MODE}", "status": "failed",
    "detail": "doctor_join_broken", "doctor_fio_pct": pct, "rows": n,
  }, ensure_ascii=False, indent=2) + "\n")
  sys.exit(3)
Path("${INBOUND}/mo_${DAY}.meta.json").write_text(
  json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(meta, ensure_ascii=False))
PY

cp -f "$EXPORT_CSV" "${INBOUND}/mo_${DAY}.csv"
if [[ -f "$EXPORT_META" ]]; then
  cp -f "$EXPORT_META" "${INBOUND}/mo_${DAY}.export.meta.json"
fi
echo "inbound ready ${INBOUND}/mo_${DAY}.csv"

FORCE_ARGS=()
if [[ "$MODE" == "retry" ]]; then
  FORCE_ARGS+=(--force)
fi
bash "$ROOT/deploy/gcp-app/score_inbound_day.sh" "$DAY" "${FORCE_ARGS[@]}"

if [[ "$WITH_LLM" == "1" ]]; then
  echo "LLM night for $DAY (background, non-fatal)"
  if sudo docker ps --format '{{.Names}}' | grep -qx protocol-web \
    && sudo docker exec protocol-web test -f /app/scripts/mo_llm_range_runner.sh; then
    sudo docker exec -d \
      -e FIRST="$DAY" -e LAST="$DAY" \
      -e SRC_ROOT=/app -e DATA="$DATA" \
      -e PYTHON=python -e RUN_HOST=gcp -e RUN_ID_PREFIX=gcp-night \
      -e MO_LLM_EXECUTION_HOST=gce \
      -e MO_ACTION_JUDGE_LIMIT="${MO_ACTION_JUDGE_LIMIT:-0}" \
      protocol-web bash /app/scripts/mo_llm_range_runner.sh \
      || echo "LLM start failed (non-fatal)"
  else
    echo "LLM runner not in container; score done. Manual: deploy/gcp-llm/run_on_gce.sh $DAY"
  fi
fi

write_status "success" "extract_score_ok"
echo "NIGHT_OK mode=${MODE} day=${DAY}"
