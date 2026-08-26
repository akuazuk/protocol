#!/usr/bin/env bash
# Night MIS extract + score on GCE (E2). Mac is not used for SQL.
#
# Modes:
#   main   - always run (cron 02:00 server/UTC)
#   retry  - only if yesterday main did not succeed (cron 03:00 = +1h)
#
# Speed / alerts (plan 2026-08-10-mo-night-speed-skip-alerts-v1):
#   MO_DAILY_WORKERS=2 (default)
#   skip --force / early exit when inbound sha256 unchanged
#   Telegram / ALERT_NEEDED after retry fail
#
# On VM:
#   bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh main
#   bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh retry
set -euo pipefail

MODE="${1:-main}"
ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
ENV_MIS="${ENV_MIS_REMOTE:-/opt/protocol/.env.mis}"
ENV_WEB="${ENV_WEB_REMOTE:-/opt/protocol/.env.gcp-staging}"
LOAD_MIS="${ROOT}/deploy/gcp-app/load_mis_env.sh"
VENV="${MIS_VENV:-/opt/protocol/venv-mis}"
# Password from Secret Manager (kravira-db-password); .env.mis is non-secret only.
export GCP_PROJECT="${GCP_PROJECT:-protocol-home-e1}"
export MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"
export MIS_PASSWORD_SOURCE="${MIS_PASSWORD_SOURCE:-secretmanager}"
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
LAB_STATUS_FILE="${STATE_DIR}/gce_lab_${DAY}.json"
STAGING="${DATA}/staging/gce-night-${DAY}-$$"
INBOUND="${DATA}/inbound/extract"
SECURE_CSV="${DATA}/secure_cases/${Y}/${M}/mo_${DAY}.csv"
CASES_JSONL="${DATA}/secure_cases/${Y}/${M}/kz_l1_${DAY}_cases.jsonl"
LOCK="${STATE_DIR}/gce-night.lock"
WITH_LLM="${MO_NIGHT_WITH_LLM:-1}"
DB_ATTEMPTS="${MO_DB_RETRIES:-5}"
DB_DELAY="${MO_DB_RETRY_DELAY_SEC:-5}"
export MO_DAILY_WORKERS="${MO_DAILY_WORKERS:-2}"
NIGHT_FORCE="${MO_NIGHT_FORCE:-0}"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$INBOUND" 2>/dev/null || true
if [[ ! -w "$LOG_DIR" ]] || [[ ! -w "$STATE_DIR" ]]; then
  sudo mkdir -p "$LOG_DIR" "$STATE_DIR" "$INBOUND" "${DATA}/staging"
  sudo chown -R "$(whoami):$(whoami)" "$LOG_DIR" "$STATE_DIR" "$INBOUND" "${DATA}/staging"
fi
exec >>"${LOG_DIR}/gce-night-${MODE}.log" 2>&1
echo "======== NIGHT ${MODE} day=${DAY} workers=${MO_DAILY_WORKERS} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"

if [[ ! -f "$LOAD_MIS" ]]; then
  echo "ERROR: missing $LOAD_MIS" >&2
  exit 2
fi
# Self-heal env ownership: deploy SSH user may differ from cron user (pavel).
ensure_env_readable() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  if [[ -r "$path" ]]; then
    return 0
  fi
  echo "WARN: $path not readable by $(whoami); attempting chown"
  if command -v sudo >/dev/null 2>&1; then
    sudo chown "$(whoami):$(whoami)" "$path" 2>/dev/null || true
    sudo chmod 600 "$path" 2>/dev/null || true
  fi
  if [[ ! -r "$path" ]]; then
    echo "WARN: still cannot read $path; continuing (SM defaults for MIS)"
  fi
}
ensure_env_readable "$ENV_MIS"
ensure_env_readable "$ENV_WEB"
ensure_env_readable "${ENV_WEB_PUBLIC:-/opt/protocol/.env.gcp-public}"
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

load_telegram_env() {
  # Prefer already-exported; else pull names from web staging env (no MIS password print).
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
          value="${value%\"}"
          value="${value#\"}"
          value="${value%\'}"
          value="${value#\'}"
          export "$key=$value"
        fi
        ;;
    esac
  done < "$ENV_WEB"
}

alert_fail() {
  local detail="$1"
  echo "ALERT_NEEDED day=${DAY} mode=${MODE} detail=${detail}"
  load_telegram_env
  if [[ -f "$ROOT/scripts/telegram_notify.py" ]]; then
    python3 "$ROOT/scripts/telegram_notify.py" \
      "МО GCE night FAIL day=${DAY} mode=${MODE} detail=${detail} host=gce" \
      >/dev/null 2>&1 || echo "telegram notify failed or disabled"
  else
    echo "telegram_notify.py missing"
  fi
}

write_status() {
  local status="$1"
  local detail="${2:-}"
  local sha="${3:-}"
  local skipped="${4:-0}"
  INBOUND_SHA="$sha" SKIPPED_SCORE="$skipped" DETAIL="$detail" STATUS="$status" \
  DAY="$DAY" MODE="$MODE" WORKERS="$MO_DAILY_WORKERS" STATUS_FILE="$STATUS_FILE" \
  python3 - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["STATUS_FILE"])
payload = {
  "day": os.environ["DAY"],
  "mode": os.environ["MODE"],
  "status": os.environ["STATUS"],
  "detail": os.environ.get("DETAIL", ""),
  "inbound_sha256": os.environ.get("INBOUND_SHA", ""),
  "workers": int(os.environ.get("WORKERS") or 0),
  "skipped_score": os.environ.get("SKIPPED_SCORE", "0") in ("1", "true", "yes"),
  "updated_at": datetime.now(timezone.utc).isoformat(),
  "host": "gce",
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {path} status={payload['status']} sha={payload['inbound_sha256'][:12]}…")
PY
}

write_lab_status() {
  local status="$1"
  local detail="${2:-}"
  LAB_STATUS="$status" LAB_DETAIL="$detail" LAB_DAY="$DAY" \
  LAB_STATUS_FILE="$LAB_STATUS_FILE" python3 - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["LAB_STATUS_FILE"])
payload = {
  "day": os.environ["LAB_DAY"],
  "status": os.environ["LAB_STATUS"],
  "detail": os.environ.get("LAB_DETAIL", ""),
  "updated_at": datetime.now(timezone.utc).isoformat(),
  "host": "gce",
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {path} status={payload['status']}")
PY
}

fail_exit() {
  local detail="$1"
  local rc="${2:-1}"
  write_status "failed" "$detail" "${INBOUND_SHA:-}"
  if [[ "$MODE" == "retry" ]]; then
    alert_fail "$detail"
  fi
  exit "$rc"
}

if [[ "$MODE" == "retry" ]]; then
  if [[ -f "$STATUS_FILE" ]]; then
    prev="$(python3 -c "import json; print(json.load(open('${STATUS_FILE}')).get('status',''))")"
    lab_prev=""
    if [[ -f "$LAB_STATUS_FILE" ]]; then
      lab_prev="$(python3 -c "import json; print(json.load(open('${LAB_STATUS_FILE}')).get('status',''))")"
    fi
    if [[ "$prev" == "success" ]] && [[ "$lab_prev" == "success" ]] \
      && [[ "$NIGHT_FORCE" != "1" ]]; then
      echo "retry skip: day=${DAY} already success"
      exit 0
    fi
    echo "retry: previous status=${prev} lab_status=${lab_prev:-missing}"
  else
    echo "retry: no status file; will run"
  fi
elif [[ "$MODE" != "main" ]]; then
  echo "Unknown mode: $MODE (main|retry)" >&2
  exit 2
fi

# shellcheck disable=SC1090
# shellcheck disable=SC1091
source "$LOAD_MIS"
export PYTHONPATH="$ROOT"
export MO_DATA_ROOT="$DATA"
export RUN_HOST=gcp
export MO_DAILY_WORKERS

mkdir -p "$STAGING"
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
  fail_exit "export_failed_after_${DB_ATTEMPTS}" 1
fi

# doctor join sanity + write inbound meta (sha filled after copy)
set +e
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
  "workers": int("${MO_DAILY_WORKERS}"),
}
if n >= 20 and pct < 50.0:
  print("ERROR: doctor_join_broken", flush=True)
  sys.exit(3)
Path("${INBOUND}/mo_${DAY}.meta.json").write_text(
  json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(meta, ensure_ascii=False))
PY
doctor_rc=$?
set -e
if [[ "$doctor_rc" -eq 3 ]]; then
  fail_exit "doctor_join_broken" 3
fi
if [[ "$doctor_rc" -ne 0 ]]; then
  fail_exit "doctor_join_script_error" "$doctor_rc"
fi

cp -f "$EXPORT_CSV" "${INBOUND}/mo_${DAY}.csv"
if [[ -f "$EXPORT_META" ]]; then
  cp -f "$EXPORT_META" "${INBOUND}/mo_${DAY}.export.meta.json"
fi
INBOUND_SHA="$(sha256sum "${INBOUND}/mo_${DAY}.csv" | awk '{print $1}')"
printf '%s\n' "$INBOUND_SHA" > "${INBOUND}/mo_${DAY}.sha256"
# patch meta with sha
python3 - <<PY
import json
from pathlib import Path
p = Path("${INBOUND}/mo_${DAY}.meta.json")
meta = json.loads(p.read_text(encoding="utf-8"))
meta["checksum_sha256"] = "${INBOUND_SHA}"
p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("inbound ready", "${INBOUND}/mo_${DAY}.csv", "sha=", "${INBOUND_SHA}"[:16]+"…")
PY

# Лаборатория: вчера + 1 день overlap (опоздавшие строки). Host venv, не protocol-web.
LAB_FROM="$(python3 -c "from datetime import date,timedelta; d=date.fromisoformat('${DAY}'); print((d-timedelta(days=1)).isoformat())")"
echo "lab ingest ${LAB_FROM}..${NEXT} (skip-coverage)"
mkdir -p "$DATA/warehouse" 2>/dev/null || sudo mkdir -p "$DATA/warehouse"
if "${VENV}/bin/python" "$ROOT/scripts/ingest_mo_lab_from_mis_tests.py" \
    --from "$LAB_FROM" --to "$NEXT" \
    --out "$DATA/warehouse/mo_lab.sqlite" \
    --skip-coverage; then
  echo "lab ingest ok"
  write_lab_status "success" "append_ok"
else
  echo "lab ingest failed (non-fatal)"
  write_lab_status "failed" "append_failed"
fi
PREV_SHA=""
PREV_STATUS=""
if [[ -f "$STATUS_FILE" ]]; then
  PREV_SHA="$(python3 -c "import json; print(json.load(open('${STATUS_FILE}')).get('inbound_sha256') or '')")"
  PREV_STATUS="$(python3 -c "import json; print(json.load(open('${STATUS_FILE}')).get('status') or '')")"
fi

SKIP_SCORE=0
FORCE_ARGS=()
if [[ "$NIGHT_FORCE" == "1" ]]; then
  echo "MO_NIGHT_FORCE=1 → force score"
  FORCE_ARGS+=(--force)
elif [[ -n "$PREV_SHA" && "$PREV_SHA" == "$INBOUND_SHA" && -f "$CASES_JSONL" && -s "$CASES_JSONL" ]]; then
  echo "inbound sha unchanged + cases present → skip score (unchanged)"
  SKIP_SCORE=1
elif [[ "$MODE" == "retry" ]]; then
  # re-score without wipe if same secure csv exists and sha matches file we just wrote to inbound only
  if [[ -f "$SECURE_CSV" ]]; then
    SECURE_SHA="$(sha256sum "$SECURE_CSV" | awk '{print $1}')"
    if [[ "$SECURE_SHA" == "$INBOUND_SHA" && -f "$CASES_JSONL" && -s "$CASES_JSONL" ]]; then
      echo "secure csv sha matches inbound + cases → skip force score"
      SKIP_SCORE=1
    else
      echo "retry: content changed or incomplete → --force"
      FORCE_ARGS+=(--force)
    fi
  else
    echo "retry: no secure csv → --force"
    FORCE_ARGS+=(--force)
  fi
else
  echo "main: resume score (no force); workers=${MO_DAILY_WORKERS}"
fi

SCORE_RC=0
if [[ "$SKIP_SCORE" == "1" ]]; then
  write_status "success" "unchanged_skip_score" "$INBOUND_SHA" 1
  echo "NIGHT_OK mode=${MODE} day=${DAY} skipped_score=1"
  exit 0
fi

set +e
bash "$ROOT/deploy/gcp-app/score_inbound_day.sh" "$DAY" "${FORCE_ARGS[@]}"
SCORE_RC=$?
set -e
if [[ "$SCORE_RC" -ne 0 ]]; then
  fail_exit "score_failed_rc_${SCORE_RC}" "$SCORE_RC"
fi

write_llm_skip() {
  local reason="$1"
  local dir="${DATA}/secure_cases/${Y}/${M}"
  local path="${dir}/kz_l1_${DAY}_llm_skip.json"
  mkdir -p "$dir" 2>/dev/null || sudo mkdir -p "$dir"
  if [[ ! -w "$dir" ]]; then
    sudo chown "$(whoami):$(whoami)" "$dir" 2>/dev/null || true
  fi
  if ! printf '%s\n' "{\"day\":\"${DAY}\",\"reason\":\"${reason}\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >"$path" 2>/dev/null; then
    printf '%s\n' "{\"day\":\"${DAY}\",\"reason\":\"${reason}\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
      | sudo tee "$path" >/dev/null
    sudo chown "$(whoami):$(whoami)" "$path" 2>/dev/null || true
  fi
  echo "wrote llm_skip for $DAY reason=${reason}"
}

if [[ "$WITH_LLM" == "1" ]]; then
  echo "LLM night for $DAY (background, non-fatal)"
  LLM_STARTED=0
  if sudo docker ps --format '{{.Names}}' | grep -qx protocol-web \
    && sudo docker exec protocol-web test -f /app/scripts/mo_llm_range_runner.sh; then
    if sudo docker exec -d \
      -e FIRST="$DAY" -e LAST="$DAY" \
      -e SRC_ROOT=/app -e DATA="$DATA" \
      -e PYTHON=python -e RUN_HOST=gcp -e RUN_ID_PREFIX=gcp-night \
      -e MO_LLM_EXECUTION_HOST=gce \
      -e MO_ACTION_JUDGE_LIMIT="${MO_ACTION_JUDGE_LIMIT:-0}" \
      protocol-web bash /app/scripts/mo_llm_range_runner.sh; then
      LLM_STARTED=1
      rm -f "${DATA}/secure_cases/${Y}/${M}/kz_l1_${DAY}_llm_skip.json" 2>/dev/null \
        || sudo rm -f "${DATA}/secure_cases/${Y}/${M}/kz_l1_${DAY}_llm_skip.json" 2>/dev/null \
        || true
    else
      echo "LLM start failed (non-fatal)"
    fi
  else
    echo "LLM runner not in container; score done. Manual: deploy/gcp-llm/run_on_gce.sh $DAY"
  fi
  if [[ "$LLM_STARTED" != "1" ]]; then
    write_llm_skip "llm_not_started" || echo "WARN: llm_skip write failed (non-fatal)"
  fi
else
  write_llm_skip "with_llm_0" || echo "WARN: llm_skip write failed (non-fatal)"
fi

# Refresh completeness after skip / before background LLM finishes (clears stale advisory).
if sudo docker ps --format '{{.Names}}' | grep -qx protocol-web; then
  sudo docker exec \
    -e FIRST="$DAY" -e LAST="$DAY" \
    -e DATA="$DATA" \
    protocol-web python /app/scripts/recompute_mo_days.py \
      --data-root /var/data/medical_exams \
      --first-date "$DAY" --last-date "$DAY" \
      --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite \
    || echo "post-score recompute failed (non-fatal)"
fi
mkdir -p "$DATA/reports" 2>/dev/null || sudo mkdir -p "$DATA/reports"
if [[ ! -w "$DATA/reports" ]]; then
  sudo chown "$(whoami):$(whoami)" "$DATA/reports"
fi
if "${VENV}/bin/python" "$ROOT/scripts/run_mo_lab_rollout_metrics.py" \
    --data-root "$DATA" --end-date "$DAY"; then
  echo "lab rollout metrics ok"
else
  echo "lab rollout metrics failed (non-fatal)"
fi

write_status "success" "extract_score_ok" "$INBOUND_SHA" 0
echo "NIGHT_OK mode=${MODE} day=${DAY} workers=${MO_DAILY_WORKERS}"
