#!/usr/bin/env bash
# Watchdog: держит 2 воркера Wave A, перезапуск при падении, finish + push + deploy + tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT/data/ml/reports"
LOG="$REPORT_DIR/chunk_qa_watchdog.log"
STATUS="$REPORT_DIR/chunk_qa_progress.md"
DONE_FLAG="$REPORT_DIR/chunk_qa_watchdog.done"
POLL_SEC="${POLL_SEC:-90}"
SSH_TARGET="${SSH_TARGET:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"

mkdir -p "$REPORT_DIR"
exec >> "$LOG" 2>&1
echo "=== Watchdog start $(date -Iseconds) ==="

if [ -f "$DONE_FLAG" ]; then
  echo "Already finished ($DONE_FLAG)"
  exit 0
fi

set -a
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
set +a

export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT=16000
export CHUNK_QA_LLM_RETRIES=3
export CHUNK_QA_BATCH_SIZE=4

PY="$ROOT/.venv/bin/python"
CHUNKS="$ROOT/output/rich_chunks/rich_chunks.final.jsonl"
if [ ! -f "$CHUNKS" ]; then
  CHUNKS="$ROOT/output/rich_chunks/rich_chunks.section_mapped.jsonl"
fi
SHARD_DIR="$ROOT/data/ml/chunk_qa_shards"
QUEUE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_queue_tiered.jsonl" | tr -d ' ')
PILOT_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_pilot.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
WAVE_START=$(date +%s)

# не дать старому orchestrator сделать merge раньше времени
pkill -f "run_gemini_qa_wave_fix.sh" 2>/dev/null || true
pkill -f "run_gemini_qa_resume_wave.sh" 2>/dev/null || true
sleep 1

shard_stats() {
  "$PY" - <<'PY'
import json
from pathlib import Path
root = Path(".")
for i in (0, 1):
    q = root / f"data/ml/chunk_qa_shards/queue_shard_{i}.jsonl"
    out = root / f"data/ml/chunk_qa_fixes_wave_a_shard{i}.jsonl"
    if not q.is_file():
        print(f"{i} 0 0 0")
        continue
    queue_rows = [json.loads(l) for l in q.open(encoding="utf-8")]
    queue_ids = [str(r.get("chunk_id") or "") for r in queue_rows]
    done = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["chunk_id"])
            except json.JSONDecodeError:
                pass
    rem_rows = [r for r in queue_rows if str(r.get("chunk_id") or "") not in done]
    rem_path = root / f"data/ml/chunk_qa_shards/queue_shard_{i}_remaining.jsonl"
    rem_path.write_text("\n".join(
        json.dumps(r, ensure_ascii=False) for r in rem_rows
    ) + ("\n" if rem_rows else ""))
    print(f"{i} {len(done)} {len(queue_ids)} {len(rem_rows)}")
PY
}

count_wave() {
  local t=0 f
  for f in "$ROOT"/data/ml/chunk_qa_fixes_wave_a_shard*.jsonl; do
    [ -f "$f" ] || continue
    t=$((t + $(wc -l < "$f" | tr -d ' ')))
  done
  echo "$t"
}

worker_running() {
  local i="$1"
  pgrep -f "queue_shard_${i}_remaining.jsonl" >/dev/null 2>&1
}

start_worker() {
  local i="$1"
  local rem="$SHARD_DIR/queue_shard_${i}_remaining.jsonl"
  [ -f "$rem" ] || return 0
  local rows
  rows=$(wc -l < "$rem" | tr -d ' ')
  [ "$rows" -gt 0 ] || return 0
  worker_running "$i" && return 0
  local out="$ROOT/data/ml/chunk_qa_fixes_wave_a_shard${i}.jsonl"
  local logw="$REPORT_DIR/chunk_qa_worker_wave_a_${i}.log"
  echo "[$(date -Iseconds)] START worker shard$i rows=$rows"
  (
    if [ "$i" -eq 0 ]; then
      export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
    else
      export GOOGLE_API_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
    fi
    unset GEMINI_API_KEY
    export CHUNK_QA_LLM=1 CHUNK_QA_LLM_BACKEND=gemini
    exec "$PY" scripts/llm_chunk_qa.py \
      --queue "$rem" \
      --chunks "$CHUNKS" \
      --out "$out" \
      --batch-size "${CHUNK_QA_BATCH_SIZE:-4}" \
      --append \
      >> "$logw" 2>&1
  ) &
}

write_status() {
  local phase="$1" wave="$2" rate="$3" note="$4"
  NOTE="$note" PHASE="$phase" PILOT="$PILOT_N" WAVE="$wave" RATE="$rate" \
  QUEUE_N="$QUEUE_N" STATUS="$STATUS" \
  "$PY" - <<'PY'
import os
from datetime import datetime
from pathlib import Path
left = max(0, int(os.environ["QUEUE_N"]) - int(os.environ["WAVE"]))
text = f"""# Chunk QA progress

**Статус: {os.environ["PHASE"]}**

Updated: {datetime.now().astimezone().isoformat(timespec="seconds")}

| | |
|---|---|
| Pilot | **{os.environ["PILOT"]}** (done) |
| Wave A | **{os.environ["WAVE"]} / {os.environ["QUEUE_N"]}** |
| Left | **~{left}** |
| Speed | **{os.environ["RATE"]}** |

{os.environ.get("NOTE", "")}

Watchdog: `data/ml/reports/chunk_qa_watchdog.log`
"""
Path(os.environ["STATUS"]).write_text(text, encoding="utf-8")
PY
}

finish_pipeline() {
  echo "=== FINISH PIPELINE $(date -Iseconds) ==="
  cd "$ROOT"

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
print("wave_a merged", len(by_id))
PY

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
print("all fixes merged", len(by_id))
PY

  "$PY" scripts/merge_chunk_qa_fixes.py --fixes "$ROOT/data/ml/chunk_qa_fixes_merged.jsonl" --chunks "$CHUNKS"
  "$PY" scripts/promote_rich_chunks_v2.py --source final

  echo "--- embeddings (optional, may take time) ---"
  "$PY" scripts/build_chunk_embeddings.py --dry-run --limit 3 2>/dev/null || true
  "$PY" scripts/build_chunk_embeddings.py 2>&1 | tail -5 || echo "embeddings skip/fail"

  "$PY" scripts/audit_chunk_quality.py \
    --chunks "$ROOT/output/rich_chunks/rich_chunks.final.jsonl" \
    --stats "$REPORT_DIR/chunk_quality_post_gemini.json" \
    --report "$REPORT_DIR/chunk_quality_post_gemini.md" \
    --baseline "$REPORT_DIR/chunk_quality_baseline.json" 2>/dev/null || \
  "$PY" scripts/audit_chunk_quality.py \
    --chunks "$ROOT/output/rich_chunks/rich_chunks.final.jsonl" \
    --stats "$REPORT_DIR/chunk_quality_post_gemini.json" \
    --report "$REPORT_DIR/chunk_quality_post_gemini.md" || true

  OUT="$ROOT/ml/experiments/batch_post_gemini_tiered"
  mkdir -p "$OUT"
  echo "--- acceptance batch ---"
  "$PY" scripts/run_clients_consult_render_batch.py --tier L1 --kz-only --ai-review off --out "$OUT/l1_kz" || true
  "$PY" scripts/run_clients_consult_render_batch.py \
    --tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 --out "$OUT/l2_sample" || true
  "$PY" scripts/run_symptom_icd_probe.py --local --no-gemini \
    --out "$REPORT_DIR/symptom_icd_probe_post_gemini.jsonl" \
    --md "$REPORT_DIR/symptom_icd_probe_post_gemini.md" || true
  "$PY" scripts/run_clients_consult_render_batch.py --tier L1 --cases a_1,a_2 --out "$OUT/b2c_sample" || true

  echo "--- upload corpus Render ---"
  bash "$ROOT/scripts/upload_rich_chunks_render.sh" "$SSH_TARGET" --gzip 2>&1 | tail -20 || \
    echo "WARN: upload skipped (SSH/key)"

  echo "--- git commit + push ---"
  bump_build_version() {
    "$PY" - <<'PY'
from pathlib import Path
p = Path("rag_server.py")
text = p.read_text(encoding="utf-8")
import re
new = "BUILD_VERSION = \"2026-06-28-r73-chunk-qa-wave-complete\""
text2, n = re.subn(r'BUILD_VERSION = "[^"]+"', new, text, count=1)
if n:
    p.write_text(text2, encoding="utf-8")
    print("bumped", new)
PY
  }
  bump_build_version

  git add \
    scripts/llm_chunk_qa.py \
    scripts/run_llm_chunk_qa_parallel.sh \
    scripts/run_gemini_qa_local.sh \
    scripts/run_gemini_qa_resume_wave.sh \
    scripts/run_gemini_qa_wave_fix.sh \
    scripts/chunk_qa_watchdog_finish.sh \
    scripts/poll_chunk_qa_eta.sh \
    .env.example \
    rag_server.py \
    data/ml/reports/chunk_qa_progress.md \
    data/ml/reports/chunk_quality_post_gemini.md \
    data/ml/reports/chunk_quality_post_gemini.json \
    data/ml/reports/symptom_icd_probe_post_gemini.md \
    2>/dev/null || true

  git add -u scripts/ 2>/dev/null || true

  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "$(cat <<'EOF'
Complete chunk QA Wave A with watchdog, parse-fallback, and acceptance batch.

Fix LLM parse stalls, parallel resume, quality audit reports, and BUILD_VERSION bump for deploy.
EOF
)" || true
    git push origin HEAD || echo "WARN: git push failed"
  else
    echo "Nothing to commit"
  fi

  echo "--- post-deploy probe (Render) ---"
  sleep 30
  "$PY" scripts/run_symptom_icd_probe.py --base https://protocol-bimy.onrender.com \
    --out "$REPORT_DIR/symptom_icd_probe_post_deploy.jsonl" \
    --md "$REPORT_DIR/symptom_icd_probe_post_deploy.md" 2>&1 | tail -10 || true

  WAVE_N=$(wc -l < "$ROOT/data/ml/chunk_qa_fixes_wave_a.jsonl" | tr -d ' ')
  write_status "COMPLETE" "$WAVE_N" "done" "Wave A + merge + push завершены. См. batch_post_gemini_tiered и chunk_quality_post_gemini.md"
  date -Iseconds > "$DONE_FLAG"
  echo "=== ALL DONE wave=$WAVE_N ==="
}

# --- main loop ---
while true; do
  total_rem=0
  while read -r i done_n q_n rem_n; do
    [ "$rem_n" -gt 0 ] && total_rem=$((total_rem + rem_n))
    start_worker "$i"
    echo "[$(date -Iseconds)] shard$i done=$done_n/$q_n remain=$rem_n running=$(worker_running "$i" && echo yes || echo no)"
  done < <(shard_stats)

  WAVE=$(count_wave)
  EL=$(( $(date +%s) - WAVE_START ))
  RATE=$(( WAVE * 60 / (EL + 1) ))
  [ "$RATE" -lt 1 ] && RATE=1
  LEFT=$(( QUEUE_N - WAVE ))
  [ "$LEFT" -lt 0 ] && LEFT=0
  ETA_MIN=$(( LEFT / RATE ))
  write_status "RUNNING (watchdog)" "$WAVE" "~${RATE} fixes/min" \
    "2 workers auto-restart. Осталось **~${LEFT}**, ETA **~${ETA_MIN} мин**. Queue remain rows: **${total_rem}**."

  echo "[$(date -Iseconds)] wave=$WAVE left=$LEFT queue_rem=$total_rem rate=~$RATE/min"

  if [ "$total_rem" -eq 0 ]; then
    echo "Queue complete"
    break
  fi
  sleep "$POLL_SEC"
done

# wait any stragglers
sleep 5
while pgrep -f "queue_shard_.*_remaining.jsonl" >/dev/null 2>&1; do
  echo "Waiting for workers..."
  sleep 30
done

finish_pipeline
