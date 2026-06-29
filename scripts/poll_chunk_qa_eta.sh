#!/usr/bin/env bash
# Обновляет chunk_qa_progress.md + chunk_qa_eta.log каждые POLL_SEC (default 120).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/data/ml/reports/chunk_qa_progress.md"
ETA_LOG="$ROOT/data/ml/reports/chunk_qa_eta.log"
POLL_SEC="${POLL_SEC:-120}"
PY="$ROOT/.venv/bin/python"

count_wave() {
  local total=0 f n
  for f in "$ROOT"/data/ml/chunk_qa_fixes_wave_a_shard*.jsonl; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f" | tr -d ' ')
    total=$((total + n))
  done
  if [ -f "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" ]; then
    n=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" | tr -d ' ')
    [ "$n" -gt "$total" ] && total=$n
  fi
  echo "$total"
}

WAVE_START=$(date +%s)
if [ -f "$ROOT/data/ml/reports/gemini_qa_resume_wave.log" ]; then
  _ts=$(grep -m1 '^=== Resume Wave A' "$ROOT/data/ml/reports/gemini_qa_resume_wave.log" | sed 's/.*Resume Wave A //' | tr -d ' ')
  if [ -n "$_ts" ]; then
    _epoch=$("$PY" -c "from datetime import datetime; print(int(datetime.fromisoformat('$_ts').timestamp()))" 2>/dev/null || true)
    [ -n "${_epoch:-}" ] && WAVE_START="$_epoch"
  fi
fi
QUEUE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_queue_tiered.jsonl" | tr -d ' ')
PILOT_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
MERGE_MIN="${MERGE_MIN:-45}"

while pgrep -f "llm_chunk_qa.py.*wave_a_shard" >/dev/null 2>&1 || \
      pgrep -f "run_llm_chunk_qa_parallel.sh wave_a" >/dev/null 2>&1; do
  WAVE=$(count_wave)
  EL=$(( $(date +%s) - WAVE_START ))
  RATE=$(( WAVE * 60 / (EL + 1) ))
  [ "$RATE" -gt 30 ] && RATE=30
  [ "$RATE" -lt 1 ] && RATE=1
  LEFT=$(( QUEUE_N - WAVE ))
  [ "$LEFT" -lt 0 ] && LEFT=0
  WAVE_MIN=$(( LEFT / RATE ))
  FINISH_EPOCH=$(( $(date +%s) + WAVE_MIN * 60 + MERGE_MIN * 60 ))

  NOTE="Wave A parallel (**2** keys). Осталось ~**${LEFT}** fixes, Wave ~**$(( WAVE_MIN / 60 ))ч $(( WAVE_MIN % 60 ))м**, полный прогон ~**$(date -r "$FINISH_EPOCH" '+%H:%M %d.%m')**."

  NOTE="$NOTE" PHASE="wave_a" PILOT="$PILOT_N" WAVE="$WAVE" RATE="~${RATE} fixes/min" \
  QUEUE_N="$QUEUE_N" LEFT="$LEFT" WAVE_MIN="$WAVE_MIN" MERGE_MIN="$MERGE_MIN" \
  FINISH="$(date -r "$FINISH_EPOCH" '+%Y-%m-%d %H:%M %z')" STATUS="$REPORT" \
  "$PY" - <<'PY'
import os
from datetime import datetime
from pathlib import Path

status = Path(os.environ["STATUS"])
note = os.environ.get("NOTE", "")
text = f"""# Chunk QA progress (Gemini local)

Updated: {datetime.now().astimezone().isoformat(timespec="seconds")}

| Metric | Value |
|--------|------:|
| Queue total | **{os.environ["QUEUE_N"]}** |
| Pilot fixes | **{os.environ["PILOT"]}** (done) |
| Wave A fixes | **{os.environ["WAVE"]}** |
| Wave A left | **~{os.environ["LEFT"]}** |
| Speed | **{os.environ["RATE"]}** |
| Wave ETA | **~{int(os.environ["WAVE_MIN"]) // 60}ч {int(os.environ["WAVE_MIN"]) % 60}м** |
| Full finish | **{os.environ["FINISH"]}** (merge+batch ~{os.environ["MERGE_MIN"]}м) |
| Phase | **{os.environ["PHASE"]}** |
| Workers | **2 keys** |

{note}

Log: `data/ml/reports/gemini_qa_resume_wave.log` · ETA log: `data/ml/reports/chunk_qa_eta.log`
"""
status.write_text(text, encoding="utf-8")
PY

  echo "[$(date -Iseconds)] wave=$WAVE left=$LEFT rate=~${RATE}/min finish~$(date -r "$FINISH_EPOCH" '+%H:%M')" >> "$ETA_LOG"
  sleep "$POLL_SEC"
done

echo "[$(date -Iseconds)] workers stopped, poll exit" >> "$ETA_LOG"
