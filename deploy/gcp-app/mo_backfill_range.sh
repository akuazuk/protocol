#!/usr/bin/env bash
# Backfill МО за диапазон дат (волна C плана 2026-09-26): для каждого дня, от
# последнего к первому, - export КЗ из МИС (venv-mis) -> inbound -> score внутри
# protocol-web (тот же путь, что ночной score_inbound_day.sh, с nice) ->
# recompute витрины за день (--skip-reports) -> каталог визитов МИС за день.
#
# Resume-safe: день с маркером state/mo_backfill_done_<день> пропускается.
# Пауза, пока идёт ночной конвейер (01:00-04:45 UTC или держится gce-night.lock).
# Один writer: flock на state/mo-backfill.lock.
#
# На VM (фон, через nohup или systemd-run):
#   nohup bash /opt/protocol/deploy/gcp-app/mo_backfill_range.sh 2026-01-01 2026-06-30 \
#     >> /var/data/medical_exams/logs/gce-mo-backfill.log 2>&1 &
# Один день для проверки:
#   bash deploy/gcp-app/mo_backfill_range.sh 2026-06-30 2026-06-30
# Остановить мягко (после текущего дня): touch /var/data/medical_exams/state/mo_backfill_stop
set -euo pipefail
FROM="${1:?YYYY-MM-DD}"
TO="${2:?YYYY-MM-DD}"
ROOT="${PROTOCOL_ROOT:-/opt/protocol}"
DATA="${GCE_MO_DATA_ROOT:-/var/data/medical_exams}"
VENV="${MIS_VENV:-/opt/protocol/venv-mis}"
CONTAINER="${MO_WEB_CONTAINER:-protocol-web}"
WORKERS="${MO_BACKFILL_WORKERS:-2}"
NICE="${MO_BACKFILL_NICE:-15}"
DB_ATTEMPTS="${MO_DB_RETRIES:-4}"
DB_DELAY="${MO_DB_RETRY_DELAY_SEC:-10}"
export GCP_PROJECT="${GCP_PROJECT:-protocol-home-e1}"
export MIS_SM_SECRET="${MIS_SM_SECRET:-kravira-db-password}"
export MIS_PASSWORD_SOURCE="${MIS_PASSWORD_SOURCE:-secretmanager}"
STATE_DIR="$DATA/state"
INBOUND="$DATA/inbound/extract"
STAGING="$DATA/staging/backfill-$$"
LOCK="$STATE_DIR/mo-backfill.lock"
NIGHT_LOCK="$STATE_DIR/gce-night.lock"
STOP_FLAG="$STATE_DIR/mo_backfill_stop"
PROGRESS="$STATE_DIR/mo_backfill_range.json"
mkdir -p "$STATE_DIR" "$INBOUND" "$STAGING" "$DATA/logs"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) backfill $*"; }

exec 9>"$LOCK"
if ! flock -n 9; then
  log "already running (lock $LOCK)"
  exit 0
fi

# shellcheck disable=SC1090
source "$ROOT/deploy/gcp-app/load_mis_env.sh"

in_night_window() {
  local hm
  hm="$((10#$(date -u +%H%M)))"
  [[ "$hm" -ge 100 && "$hm" -lt 445 ]]
}

night_lock_held() {
  # gce-night.lock - PID-файл ночного конвейера (noclobber), не flock.
  [[ -f "$NIGHT_LOCK" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$NIGHT_LOCK" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_window() {
  while in_night_window || night_lock_held; do
    log "night pipeline window/lock - sleeping 10 min"
    sleep 600
  done
}

write_progress() {
  python3 - "$PROGRESS" "$1" "$2" "$3" <<'PY'
import json, sys
from datetime import datetime, timezone
path, day, status, note = sys.argv[1:5]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    data = {"days": {}}
data.setdefault("days", {})[day] = {
    "status": status,
    "note": note,
    "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
}

DAYS="$(python3 - "$FROM" "$TO" <<'PY'
import sys
from datetime import date, timedelta
a, b = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
if a > b:
    a, b = b, a
d = b
while d >= a:
    print(d.isoformat())
    d -= timedelta(days=1)
PY
)"

for DAY in $DAYS; do
  if [[ -f "$STOP_FLAG" ]]; then
    log "stop flag present - exiting before $DAY"
    rm -f "$STOP_FLAG"
    exit 0
  fi
  Y="${DAY:0:4}"; M="${DAY:5:2}"
  DONE_MARK="$STATE_DIR/mo_backfill_done_${DAY}"
  if [[ -f "$DONE_MARK" ]]; then
    continue
  fi
  wait_for_window
  NEXT="$(python3 -c "from datetime import date,timedelta; print((date.fromisoformat('${DAY}')+timedelta(days=1)).isoformat())")"
  log "day=$DAY export"
  EXPORT_CSV="$STAGING/mis_protocol_${DAY}_${NEXT}.csv"
  export_ok=0
  attempt=1
  while [[ "$attempt" -le "$DB_ATTEMPTS" ]]; do
    if "$VENV/bin/python" "$ROOT/scripts/export_mis_protocol_month.py" \
        --from "$DAY" --to "$NEXT" --out-dir "$STAGING" >/dev/null 2>&1; then
      export_ok=1
      break
    fi
    log "day=$DAY export failed attempt=$attempt"
    sleep $(( DB_DELAY * attempt ))
    attempt=$((attempt + 1))
  done
  if [[ "$export_ok" != "1" || ! -f "$EXPORT_CSV" ]]; then
    write_progress "$DAY" "export_failed" ""
    continue
  fi
  ROWS="$(python3 -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open('${EXPORT_CSV}', encoding='utf-8'))))")"
  cp -f "$EXPORT_CSV" "$INBOUND/mo_${DAY}.csv"
  SHA="$(sha256sum "$INBOUND/mo_${DAY}.csv" | awk '{print $1}')"
  printf '%s\n' "$SHA" > "$INBOUND/mo_${DAY}.sha256"
  python3 - "$INBOUND/mo_${DAY}.meta.json" "$DAY" "$ROWS" "$SHA" "$WORKERS" <<'PY'
import json, sys
from datetime import datetime, timezone
path, day, rows, sha, workers = sys.argv[1:6]
json.dump({
  "schema_version": 1, "day": day, "row_count": int(rows), "checksum_sha256": sha,
  "extracted_at": datetime.now(timezone.utc).isoformat(), "run_host": "gcp",
  "source": "kravira_mc.mis_protocol + mis_data", "mode": "backfill", "workers": int(workers),
}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
  log "day=$DAY rows=$ROWS score (workers=$WORKERS nice=$NICE)"
  set +e
  sudo docker exec \
    -e MO_DATA_ROOT="$DATA" -e DAY="$DAY" -e Y="$Y" -e M="$M" \
    -e MO_DAILY_WORKERS="$WORKERS" -e RUN_HOST=gcp -e NICE_LEVEL="$NICE" \
    "$CONTAINER" bash -lc '
set -euo pipefail
DATA="${MO_DATA_ROOT}"
SECURE="$DATA/secure_cases/${Y}/${M}"
mkdir -p "$SECURE"
cp -f "$DATA/inbound/extract/mo_${DAY}.csv" "$SECURE/mo_${DAY}.csv"
cp -f "$DATA/inbound/extract/mo_${DAY}.meta.json" "$SECURE/mo_${DAY}.meta.json" 2>/dev/null || true
nice -n "$NICE_LEVEL" python scripts/run_mis_protocol_l1_batch.py \
  --csv "$SECURE/mo_${DAY}.csv" --out-dir "$SECURE" --month "$DAY" \
  --direct --deep-eval --resume --workers "$MO_DAILY_WORKERS"
nice -n "$NICE_LEVEL" python scripts/recompute_mo_days.py \
  --data-root "$DATA" --first-date "$DAY" --last-date "$DAY" \
  --warehouse "$DATA/warehouse/mo_analytics.sqlite" --skip-reports >/dev/null
echo BACKFILL_DAY_OK
'
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "day=$DAY score failed rc=$rc (will retry on next run)"
    write_progress "$DAY" "score_failed" "rc=$rc"
    continue
  fi
  if "$VENV/bin/python" "$ROOT/scripts/ingest_mo_mis_catalog.py" \
      --from "$DAY" --to "$NEXT" --warehouse "$DATA/warehouse/mo_analytics.sqlite" >/dev/null 2>&1; then
    :
  else
    log "day=$DAY mis catalog ingest failed (non-fatal)"
  fi
  touch "$DONE_MARK"
  write_progress "$DAY" "done" "rows=$ROWS"
  log "day=$DAY done"
done
rm -rf "$STAGING"
log "range $FROM..$TO finished"
