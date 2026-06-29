#!/usr/bin/env bash
# Доработка Wave A: remaining queue + 2 воркера (--append), merge, promote, batch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT/data/ml/reports"
LOG="$REPORT_DIR/gemini_qa_wave_fix.log"
STATUS="$REPORT_DIR/chunk_qa_progress.md"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p "$REPORT_DIR"
exec >> "$LOG" 2>&1
echo "=== Wave A fix/resume $(date -Iseconds) ==="

set -a
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
set +a

export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT=16000
export CHUNK_QA_LLM_RETRIES=3
export CHUNK_QA_BATCH_SIZE=4

PY="$ROOT/.venv/bin/python"
CHUNKS="$ROOT/output/rich_chunks/rich_chunks.section_mapped.jsonl"
SHARD_DIR="$ROOT/data/ml/chunk_qa_shards"
QUEUE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_queue_tiered.jsonl" | tr -d ' ')
PILOT_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" 2>/dev/null | tr -d ' ' || echo 0)

# stop stuck workers
pkill -f "llm_chunk_qa.py.*wave_a_shard" 2>/dev/null || true
pkill -f "run_llm_chunk_qa_parallel.sh wave_a" 2>/dev/null || true
pkill -f "run_gemini_qa_resume_wave.sh" 2>/dev/null || true
sleep 2

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path(".")
shard_dir = root / "data/ml/chunk_qa_shards"
shard_dir.mkdir(parents=True, exist_ok=True)
for i in (0, 1):
    q = shard_dir / f"queue_shard_{i}.jsonl"
    out = root / f"data/ml/chunk_qa_fixes_wave_a_shard{i}.jsonl"
    if not q.is_file():
        continue
    queue_rows = [json.loads(l) for l in q.open(encoding="utf-8")]
    done = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(str(json.loads(line).get("chunk_id") or ""))
            except json.JSONDecodeError:
                pass
    remain = [r for r in queue_rows if str(r.get("chunk_id") or "") not in done]
    rem_path = shard_dir / f"queue_shard_{i}_remaining.jsonl"
    rem_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in remain) + ("\n" if remain else ""))
    print(json.dumps({"shard": i, "done": len(done), "remain": len(remain)}, ensure_ascii=False))
PY

collect_keys() {
  KEYS=()
  if [ -n "${CHUNK_QA_GOOGLE_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$CHUNK_QA_GOOGLE_API_KEYS"
  elif [ -n "${GOOGLE_API_KEY:-}" ]; then
    KEYS+=("$GOOGLE_API_KEY")
    [ -n "${GEMINI_API_KEY:-}" ] && [ "$GEMINI_API_KEY" != "$GOOGLE_API_KEY" ] && KEYS+=("$GEMINI_API_KEY")
  fi
  echo "${#KEYS[@]}"
}

KEYS_N=$(collect_keys)
[ "$KEYS_N" -ge 1 ] || { echo "No API keys"; exit 1; }

count_wave() {
  local t=0 f n
  for f in "$ROOT"/data/ml/chunk_qa_fixes_wave_a_shard*.jsonl; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f" | tr -d ' ')
    t=$((t + n))
  done
  echo "$t"
}

write_report() {
  local phase="$1" wave="$2" rate="$3" note="$4"
  NOTE="$note" PHASE="$phase" PILOT="$PILOT_N" WAVE="$wave" RATE="$rate" QUEUE_N="$QUEUE_N" STATUS="$STATUS" \
  "$PY" - <<'PY'
import os
from datetime import datetime
from pathlib import Path
left = max(0, int(os.environ["QUEUE_N"]) - int(os.environ["WAVE"]))
text = f"""# Chunk QA progress

**Статус: {os.environ["PHASE"].upper()}**

Updated: {datetime.now().astimezone().isoformat(timespec="seconds")}

| | |
|---|---|
| Pilot | **{os.environ["PILOT"]}** fixes (done) |
| Wave A | **{os.environ["WAVE"]} / {os.environ["QUEUE_N"]}** |
| Left | **~{left}** |
| Speed | **{os.environ["RATE"]}** |

{os.environ.get("NOTE", "")}

Log: `data/ml/reports/gemini_qa_wave_fix.log`
"""
Path(os.environ["STATUS"]).write_text(text, encoding="utf-8")
PY
}

WAVE_START=$(date +%s)
PIDS=()
for i in 0 1; do
  REM="$SHARD_DIR/queue_shard_${i}_remaining.jsonl"
  [ -f "$REM" ] || continue
  ROWS=$(wc -l < "$REM" | tr -d ' ')
  [ "$ROWS" -eq 0 ] && continue
  if [ "$i" -eq 0 ]; then
    export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
    unset GEMINI_API_KEY
  else
    export GOOGLE_API_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
    unset GEMINI_API_KEY
  fi
  OUT="$ROOT/data/ml/chunk_qa_fixes_wave_a_shard${i}.jsonl"
  LOGW="$REPORT_DIR/chunk_qa_worker_wave_a_${i}.log"
  echo "Resume worker $i: $ROWS remaining"
  (
    export CHUNK_QA_LLM=1 CHUNK_QA_LLM_BACKEND=gemini
    exec "$PY" scripts/llm_chunk_qa.py \
      --queue "$REM" \
      --chunks "$CHUNKS" \
      --out "$OUT" \
      --batch-size "${CHUNK_QA_BATCH_SIZE:-4}" \
      --append \
      >> "$LOGW" 2>&1
  ) &
  PIDS+=($!)
done

write_report "wave_a resume" "$(count_wave)" "..." "Wave A resume: **2** workers, batch-size **4**"

while [ "${#PIDS[@]}" -gt 0 ]; do
  ALIVE=()
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && ALIVE+=("$pid")
  done
  PIDS=("${ALIVE[@]}")
  WAVE=$(count_wave)
  EL=$(( $(date +%s) - WAVE_START ))
  RATE=$(( WAVE * 60 / (EL + 1) ))
  LEFT=$(( QUEUE_N - WAVE ))
  ETA_MIN=$(( LEFT / (RATE + 1) ))
  write_report "wave_a resume" "$WAVE" "~${RATE} fixes/min" "Осталось **~${LEFT}**, ETA Wave **~${ETA_MIN} мин**"
  echo "[$(date -Iseconds)] wave=$WAVE left=$LEFT rate=~${RATE}/min"
  [ "${#PIDS[@]}" -eq 0 ] && break
  sleep "$POLL_SEC"
done

for pid in "${PIDS[@]}"; do wait "$pid" || true; done

echo "--- merge shards ---"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path(".")
by_id = {}
for fp in sorted(root.glob("data/ml/chunk_qa_fixes_wave_a_shard*.jsonl")):
    for line in fp.open(encoding="utf-8"):
        r = json.loads(line)
        cid = str(r.get("chunk_id") or "")
        if cid:
            by_id[cid] = r
out = root / "data/ml/chunk_qa_fixes_wave_a.jsonl"
out.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged wave_a", len(by_id))
PY

WAVE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" | tr -d ' ')
echo "--- Merge + promote ---"
cd "$ROOT"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path(".")
by_id = {}
for fp in ["data/ml/chunk_qa_fixes_merged.jsonl", "data/ml/chunk_qa_fixes_pilot.jsonl", "data/ml/chunk_qa_fixes_wave_a.jsonl"]:
    p = root / fp
    if not p.is_file():
        continue
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        cid = str(r.get("chunk_id") or "")
        if cid:
            by_id[cid] = r
(root / "data/ml/chunk_qa_fixes_merged.jsonl").write_text(
    "\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged total", len(by_id))
PY

"$PY" scripts/merge_chunk_qa_fixes.py --fixes "$ROOT/data/ml/chunk_qa_fixes_merged.jsonl" --chunks "$CHUNKS"
"$PY" scripts/promote_rich_chunks_v2.py --source final

OUT="$ROOT/ml/experiments/batch_post_gemini_tiered"
mkdir -p "$OUT"
"$PY" scripts/run_clients_consult_render_batch.py --tier L1 --kz-only --ai-review off --out "$OUT/l1_kz" || true
"$PY" scripts/run_clients_consult_render_batch.py \
  --tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 --out "$OUT/l2_sample" || true

write_report "complete" "$WAVE_N" "done" "Wave A + merge + batch **готово**"
echo "=== DONE wave=$WAVE_N ==="
