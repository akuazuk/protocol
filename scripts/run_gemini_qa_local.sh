#!/usr/bin/env bash
# Локальный chunk QA через Gemini API (самый быстрый путь при наличии GOOGLE_API_KEY).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT/data/ml/reports"
LOG="$REPORT_DIR/gemini_qa_local_run.log"
STATUS="$REPORT_DIR/chunk_qa_progress.md"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p "$REPORT_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "=== Local Gemini QA $(date -Iseconds) ==="

set -a
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
set +a

export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT=16000
export CHUNK_QA_LLM_RETRIES=5

count_gemini_keys() {
  local n=0
  if [ -n "${CHUNK_QA_GOOGLE_API_KEYS:-}" ]; then
    IFS=',' read -ra _k <<< "$CHUNK_QA_GOOGLE_API_KEYS"
    n="${#_k[@]}"
  elif [ -n "${GOOGLE_API_KEY:-}" ]; then
    n=1
    if [ -n "${GEMINI_API_KEY:-}" ] && [ "$GEMINI_API_KEY" != "$GOOGLE_API_KEY" ]; then
      n=$((n + 1))
    fi
    for i in 2 3 4 5; do
      v="GOOGLE_API_KEY_${i}"
      [ -n "${!v:-}" ] && n=$((n + 1))
    done
  elif [ -n "${GEMINI_API_KEY:-}" ]; then
    n=1
  fi
  echo "$n"
}

count_wave_shard_fixes() {
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

GEMINI_KEYS_N=$(count_gemini_keys)
echo "Gemini API keys detected: $GEMINI_KEYS_N"

PY="$ROOT/.venv/bin/python"
QUEUE="$ROOT/data/ml/chunk_qa_queue_tiered.jsonl"
CHUNKS="$ROOT/output/rich_chunks/rich_chunks.section_mapped.jsonl"
CHUNKS_FINAL="$ROOT/output/rich_chunks/rich_chunks.final.jsonl"

if [ ! -f "$QUEUE" ] || [ ! -f "$CHUNKS" ]; then
  "$PY" scripts/build_chunk_qa_queue_tiered.py --chunks "$CHUNKS_FINAL" \
    --fixes "$ROOT/data/ml/chunk_qa_fixes_merged.jsonl" --kz-folder clients_consult \
    --out "$QUEUE" --manifest "$ROOT/data/ml/chunk_qa_queue_tiered_manifest.json"
  "$PY" scripts/apply_chunk_rule_fixes.py --in "$CHUNKS_FINAL" --out "$ROOT/output/rich_chunks/rich_chunks.rules.jsonl"
  "$PY" scripts/apply_protocol_section_map.py --in "$ROOT/output/rich_chunks/rich_chunks.rules.jsonl" --out "$CHUNKS"
fi

QUEUE_N=$(wc -l < "$QUEUE" | tr -d ' ')
START_TS=$(date +%s)

write_report() {
  local phase="$1" pilot="$2" wave="$3" rate="$4" note="${5:-}"
  NOTE="$note" PHASE="$phase" PILOT="$pilot" WAVE="$wave" RATE="$rate" QUEUE_N="$QUEUE_N" STATUS="$STATUS" \
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
| Pilot target | **800** |
| Pilot fixes | **{os.environ["PILOT"]}** |
| Wave A fixes | **{os.environ["WAVE"]}** |
| Speed | **{os.environ["RATE"]}** |
| Phase | **{os.environ["PHASE"]}** |
| Where | **Mac local (Gemini API)** |

{note}

Log: `data/ml/reports/gemini_qa_local_run.log`
"""
status.write_text(text, encoding="utf-8")
PY
}

write_report "starting" 0 0 "..." "Pilot 800 + Wave ${QUEUE_N}"

rm -f "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl"

echo "--- Phase 1: pilot 800 ---"
"$PY" scripts/llm_chunk_qa.py --queue "$QUEUE" --chunks "$CHUNKS" \
  --out "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" --limit 800 --batch-size 8 &
PID_PILOT=$!

while kill -0 "$PID_PILOT" 2>/dev/null; do
  sleep "$POLL_SEC"
  PILOT=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  EL=$(( $(date +%s) - START_TS ))
  RATE="~$(( PILOT * 60 / (EL + 1) )) fixes/min"
  write_report "pilot" "$PILOT" 0 "$RATE" "Pilot в процессе..."
  echo "[$(date -Iseconds)] pilot=$PILOT rate=$RATE"
done
wait "$PID_PILOT"
PILOT_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" | tr -d ' ')

# Перечитать .env: ключи могли добавить во время pilot
set -a
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
set +a
export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
GEMINI_KEYS_N=$(count_gemini_keys)
echo "--- Phase 2: Wave A (${GEMINI_KEYS_N} key(s)) ---"
WAVE_START=$(date +%s)
rm -f "$ROOT"/data/ml/chunk_qa_fixes_wave_a_shard*.jsonl

if [ "$GEMINI_KEYS_N" -ge 2 ]; then
  write_report "wave_a" "$PILOT_N" 0 "..." "Wave A parallel: **${GEMINI_KEYS_N}** воркеров"
  bash "$ROOT/scripts/run_llm_chunk_qa_parallel.sh" wave_a &
  PID_WAVE=$!
  while kill -0 "$PID_WAVE" 2>/dev/null; do
    sleep "$POLL_SEC"
    WAVE=$(count_wave_shard_fixes)
    EL=$(( $(date +%s) - WAVE_START ))
    RATE="~$(( WAVE * 60 / (EL + 1) )) fixes/min"
    write_report "wave_a" "$PILOT_N" "$WAVE" "$RATE" "Wave A parallel (**${GEMINI_KEYS_N}** keys)..."
    echo "[$(date -Iseconds)] wave=$WAVE workers=$GEMINI_KEYS_N rate=$RATE"
  done
  wait "$PID_WAVE"
else
  write_report "wave_a" "$PILOT_N" 0 "..." "Wave A: 1 ключ (добавьте GOOGLE_API_KEY_2 для ускорения)"
  "$PY" scripts/llm_chunk_qa.py --queue "$QUEUE" --chunks "$CHUNKS" \
    --out "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" --batch-size 8 --append &
  PID_WAVE=$!
  while kill -0 "$PID_WAVE" 2>/dev/null; do
    sleep "$POLL_SEC"
    WAVE=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
    EL=$(( $(date +%s) - WAVE_START ))
    RATE="~$(( WAVE * 60 / (EL + 1) )) fixes/min"
    write_report "wave_a" "$PILOT_N" "$WAVE" "$RATE" "Wave A в процессе (1 key)..."
    echo "[$(date -Iseconds)] wave=$WAVE rate=$RATE"
  done
  wait "$PID_WAVE"
fi

WAVE_N=$(count_wave_shard_fixes)
if [ -f "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" ]; then
  WAVE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" | tr -d ' ')
fi

echo "--- Merge + promote ---"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path(".")
by_id = {}
for fp in ["data/ml/chunk_qa_fixes_merged.jsonl", "data/ml/chunk_qa_fixes_pilot.jsonl", "data/ml/chunk_qa_fixes_wave_a.jsonl"]:
    p = root / fp
    if not p.is_file(): continue
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        cid = str(r.get("chunk_id") or "")
        if cid: by_id[cid] = r
(root / "data/ml/chunk_qa_fixes_merged.jsonl").write_text(
    "\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged", len(by_id))
PY

"$PY" scripts/merge_chunk_qa_fixes.py --fixes "$ROOT/data/ml/chunk_qa_fixes_merged.jsonl" --chunks "$CHUNKS"
"$PY" scripts/promote_rich_chunks_v2.py --source final

OUT="$ROOT/ml/experiments/batch_post_gemini_tiered"
mkdir -p "$OUT"
"$PY" scripts/run_clients_consult_render_batch.py --tier L1 --kz-only --ai-review off --out "$OUT/l1_kz" || true
"$PY" scripts/run_clients_consult_render_batch.py \
  --tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 --out "$OUT/l2_sample" || true

write_report "complete" "$PILOT_N" "$WAVE_N" "done" "Merge + batch L1/L2 done."
echo "=== DONE pilot=$PILOT_N wave=$WAVE_N ==="
