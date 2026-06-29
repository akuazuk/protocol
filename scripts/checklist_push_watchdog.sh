#!/usr/bin/env bash
# Post-Wave-A: Telegram только этапы фоновых задач (embed, ошибки). Без commit/push/redeploy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
POLL_SEC="${POLL_SEC:-120}"
STATE="$ROOT/data/ml/reports/checklist_notify_state.json"
LOG="$ROOT/data/ml/reports/checklist_notify.log"
RICH="$ROOT/output/rich_chunks/rich_chunks.jsonl"
PROD_URL="${PROTOCOL_PROD_URL:-https://protocol-bimy.onrender.com}"
EXPECTED_BUILD="$("$PY" -c "import re; t=open('rag_server.py',encoding='utf-8').read(); m=re.search(r'BUILD_VERSION = \"([^\"]+)\"', t); print(m.group(1) if m else '')")"

mkdir -p "$(dirname "$STATE")"

render_tg_enabled() {
  [ "${TELEGRAM_NOTIFY_RENDER:-0}" = "1" ]
}

git_tg_enabled() {
  [ "${TELEGRAM_NOTIFY_GIT:-0}" = "1" ]
}

send_tg() {
  local msg="$1"
  echo "[$(date -Iseconds)] TG: $msg" >> "$LOG"
  "$PY" scripts/telegram_notify.py "$msg" 2>/dev/null || true
}

log_only() {
  echo "[$(date -Iseconds)] $1" >> "$LOG"
}

ask_decision() {
  local id="$1" text="$2" yes_action="$3" no_action="${4:-noop}" yes_label="${5:-Да}" no_label="${6:-Нет}"
  echo "[$(date -Iseconds)] ask $id" >> "$LOG"
  "$PY" scripts/telegram_control.py ask \
    --id "$id" --text "$text" --yes "$yes_action" --no "$no_action" \
    --yes-label "$yes_label" --no-label "$no_label" 2>/dev/null || send_tg "$text"
}

read_state() {
  [ -f "$STATE" ] && cat "$STATE" || echo '{}'
}

write_state_json() {
  STATE="$STATE" PHASE="${1:-}" DETAIL="${2:-}" TS="$(date -Iseconds)" EMB_MILESTONE="${3:-}" "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["STATE"])
prev = {}
if p.is_file():
    try:
        prev = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
prev.update({
    "phase": os.environ.get("PHASE", prev.get("phase", "")),
    "detail": os.environ.get("DETAIL", prev.get("detail", "")),
    "ts": os.environ.get("TS", ""),
})
em = os.environ.get("EMB_MILESTONE", "").strip()
if em:
    prev["embed_milestone_last"] = int(em)
p.write_text(json.dumps(prev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

notify_stage_once() {
  local phase="$1" detail="$2" msg="$3"
  local prev
  prev=$(read_state | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d.get('phase',''))" 2>/dev/null || echo "")
  if [ "$prev" = "$phase" ] && [ "$phase" != "embed_progress" ]; then
    return 0
  fi
  write_state_json "$phase" "$detail" ""
  send_tg "$msg"
}

notify_embed_milestone() {
  local pct="$1" done="$2" total="$3"
  local last
  last=$(read_state | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(int(d.get('embed_milestone_last') or 0))" 2>/dev/null || echo "0")
  if [ "$pct" -le "$last" ]; then
    return 0
  fi
  # уведомляем только на 25/50/75/100
  local bucket=0
  if [ "$pct" -ge 100 ]; then bucket=100
  elif [ "$pct" -ge 75 ]; then bucket=75
  elif [ "$pct" -ge 50 ]; then bucket=50
  elif [ "$pct" -ge 25 ]; then bucket=25
  else return 0
  fi
  if [ "$bucket" -le "$last" ]; then
    return 0
  fi
  write_state_json "embed_progress" "pct=$bucket done=$done/$total" "$bucket"
  send_tg "Protocol [embed]: $bucket% готово ($done / $total чанков). Ошибок: см. embed_checklist_run.log."
}

embed_running() {
  pgrep -f "scripts/build_chunk_embeddings.py" >/dev/null 2>&1
}

embed_stats() {
  STATE_JSON="$ROOT/output/embed_build_state.json" "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ.get("STATE_JSON", ""))
if not p.is_file():
    print("0 0 0")
    raise SystemExit(0)
d = json.loads(p.read_text(encoding="utf-8"))
st = d.get("stats") or {}
done = len(d.get("done_chunk_ids") or [])
errs = len(d.get("error_chunk_ids") or [])
print(int(st.get("embedded", 0)), int(st.get("errors", errs)), done)
PY
}

total_chunks() {
  if [ -f "$RICH" ]; then
    wc -l < "$RICH" | tr -d ' '
  else
    echo "57852"
  fi
}

prod_version() {
  curl -sf "${PROD_URL}/api/version" 2>/dev/null | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('version',''))" 2>/dev/null || echo ""
}

git_ahead() {
  git rev-list --count origin/main..HEAD 2>/dev/null || echo 0
}

once_check() {
  local emb emb_err done_ids total pct prod ahead
  total=$(total_chunks)
  read -r emb emb_err done_ids < <(embed_stats)
  pct=0
  if [ "${total:-0}" -gt 0 ]; then
    pct=$(( done_ids * 100 / total ))
  fi

  if embed_running; then
    notify_embed_milestone "$pct" "$done_ids" "$total"
    return 0
  fi

  # embed завершён (≥99%)
  if [ "${total:-0}" -gt 0 ] && [ "${done_ids:-0}" -ge $(( total * 99 / 100 )) ]; then
    local last
    last=$(read_state | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d.get('phase',''))" 2>/dev/null || echo "")
    if [ "$last" != "embed_complete" ]; then
      write_state_json "embed_complete" "done=$done_ids/$total" "100"
      send_tg "Protocol [embed]: завершено ($done_ids / $total). Можно upload corpus и symptom probe."
    fi
    return 0
  fi

  if [ "${emb_err:-0}" -gt 100 ]; then
    notify_stage_once "embed_errors" "errors=$emb_err" \
      "Protocol [embed]: ошибки ($emb_err). Проверьте embed_checklist_run.log и GOOGLE_API_KEY."
    return 0
  fi

  # git / prod / redeploy — только в лог, не в Telegram (если TELEGRAM_NOTIFY_GIT/RENDER выключены)
  prod=$(prod_version)
  ahead=$(git_ahead)

  if git_tg_enabled && [ "${ahead:-0}" -gt 0 ]; then
    if render_tg_enabled; then
      notify_stage_once "git_push_pending" "ahead=$ahead" \
        "Protocol: локально $ahead commit(ов) не на origin. Сделайте git push, затем redeploy Render до $EXPECTED_BUILD."
    elif [ "${TELEGRAM_INTERACTIVE:-1}" = "1" ]; then
      ask_decision "git_push" \
        "Protocol: локально $ahead commit(ов) не на origin. Сделать git push?" \
        "git_push" "git_push_skip" "Да, push" "Нет"
    else
      notify_stage_once "git_push_pending" "ahead=$ahead" \
        "Protocol: локально $ahead commit(ов) не на origin."
    fi
    return 0
  fi

  if [ "${ahead:-0}" -gt 0 ]; then
    log_only "git ahead=$ahead (telegram off TELEGRAM_NOTIFY_GIT=0)"
  fi

  if render_tg_enabled && [ -n "$EXPECTED_BUILD" ] && [ -n "$prod" ] && [ "$prod" != "$EXPECTED_BUILD" ]; then
    notify_stage_once "redeploy_pending" "prod=$prod expected=$EXPECTED_BUILD" \
      "Protocol: git на origin, prod ещё $prod. Redeploy Render → $EXPECTED_BUILD."
  elif [ -n "$EXPECTED_BUILD" ] && [ -n "$prod" ] && [ "$prod" != "$EXPECTED_BUILD" ]; then
    log_only "redeploy_pending prod=$prod expected=$EXPECTED_BUILD"
  fi

  if [ -n "$EXPECTED_BUILD" ] && [ "$prod" = "$EXPECTED_BUILD" ]; then
    log_only "prod_ready prod=$prod"
  fi
}

case "${1:-loop}" in
  once) once_check ;;
  loop)
    echo "checklist_push_watchdog start poll=${POLL_SEC}s stages-only git_tg=$(git_tg_enabled && echo 1 || echo 0)" >> "$LOG"
    while true; do
      once_check
      sleep "$POLL_SEC"
    done
    ;;
  *)
    echo "Usage: $0 [once|loop]" >&2
    exit 2
    ;;
esac
