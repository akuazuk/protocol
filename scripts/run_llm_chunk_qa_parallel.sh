#!/usr/bin/env bash
# Параллельный chunk QA: N ключей Gemini → N воркеров, merge fixes.
# Каждый ключ = отдельная квота RPM (разные проекты AI Studio).
#
# .env:
#   CHUNK_QA_GOOGLE_API_KEYS=key1,key2,key3
# или GOOGLE_API_KEY + GEMINI_API_KEY (разные проекты) + GOOGLE_API_KEY_2...
#
# Usage:
#   bash scripts/run_llm_chunk_qa_parallel.sh wave_a
#   bash scripts/run_llm_chunk_qa_parallel.sh pilot   # 800 / N на воркер
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

set -a
[ -f .env ] && source .env
set +a

export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT="${CHUNK_QA_MAX_OUT:-16000}"
export CHUNK_QA_LLM_RETRIES="${CHUNK_QA_LLM_RETRIES:-5}"

PY="$ROOT/.venv/bin/python"
QUEUE="$ROOT/data/ml/chunk_qa_queue_tiered.jsonl"
CHUNKS="$ROOT/output/rich_chunks/rich_chunks.section_mapped.jsonl"
MODE="${1:-wave_a}"
WORKERS="${CHUNK_QA_WORKERS:-0}"

# collect keys
KEYS=()
if [ -n "${CHUNK_QA_GOOGLE_API_KEYS:-}" ]; then
  IFS=',' read -ra KEYS <<< "$CHUNK_QA_GOOGLE_API_KEYS"
elif [ -n "${GOOGLE_API_KEY:-}" ]; then
  KEYS+=("$GOOGLE_API_KEY")
  if [ -n "${GEMINI_API_KEY:-}" ] && [ "$GEMINI_API_KEY" != "$GOOGLE_API_KEY" ]; then
    KEYS+=("$GEMINI_API_KEY")
  fi
  for i in 2 3 4 5; do
    v="GOOGLE_API_KEY_${i}"
    [ -n "${!v:-}" ] && KEYS+=("${!v}")
  done
elif [ -n "${GEMINI_API_KEY:-}" ]; then
  KEYS+=("$GEMINI_API_KEY")
fi

if [ "${#KEYS[@]}" -lt 1 ]; then
  echo "Нет ключей: CHUNK_QA_GOOGLE_API_KEYS или GOOGLE_API_KEY" >&2
  exit 1
fi

if [ "$WORKERS" -gt 0 ] 2>/dev/null; then
  N="$WORKERS"
else
  N="${#KEYS[@]}"
fi
N=$(( N < ${#KEYS[@]} ? N : ${#KEYS[@]} ))
N=$(( N > 1 ? N : 1 ))

echo "Parallel chunk QA: mode=$MODE workers=$N keys=${#KEYS[@]}"

SHARD_DIR="$ROOT/data/ml/chunk_qa_shards"
mkdir -p "$SHARD_DIR"

"$PY" - <<PY
import json
from collections import defaultdict
from pathlib import Path

root = Path("$ROOT")
queue = root / "data/ml/chunk_qa_queue_tiered.jsonl"
chunks_path = root / "output/rich_chunks/rich_chunks.section_mapped.jsonl"
n = int("$N")
mode = "$MODE"
shard_dir = root / "data/ml/chunk_qa_shards"
shard_dir.mkdir(parents=True, exist_ok=True)

# index doc_id per queue row
rows = [json.loads(l) for l in queue.open(encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[str(r.get("doc_id") or "unknown")].append(r)

docs = sorted(by_doc.keys())
shards = [[] for _ in range(n)]
for i, doc in enumerate(docs):
    shards[i % n].extend(by_doc[doc])

if mode == "pilot":
    cap = 800
    out_shards = []
    taken = 0
    for s in shards:
        part = []
        for r in s:
            if taken >= cap:
                break
            part.append(r)
            taken += 1
        out_shards.append(part)
    shards = out_shards

for i, s in enumerate(shards):
    p = shard_dir / f"queue_shard_{i}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in s) + ("\n" if s else ""))
    print(json.dumps({"shard": i, "rows": len(s)}, ensure_ascii=False))
PY

PIDS=()
for i in $(seq 0 $((N - 1))); do
  KEY="${KEYS[$i]}"
  OUT="$ROOT/data/ml/chunk_qa_fixes_${MODE}_shard${i}.jsonl"
  LOG="$ROOT/data/ml/reports/chunk_qa_worker_${MODE}_${i}.log"
  SHARD="$SHARD_DIR/queue_shard_${i}.jsonl"
  ROWS=$(wc -l < "$SHARD" | tr -d ' ')
  [ "$ROWS" -eq 0 ] && continue
  echo "Worker $i: $ROWS queue rows -> $OUT"
  (
    export GOOGLE_API_KEY="$KEY"
    unset GEMINI_API_KEY
    exec "$PY" scripts/llm_chunk_qa.py \
      --queue "$SHARD" \
      --chunks "$CHUNKS" \
      --out "$OUT" \
      --batch-size "${CHUNK_QA_BATCH_SIZE:-8}" \
      >> "$LOG" 2>&1
  ) &
  PIDS+=($!)
done

echo "Waiting for workers: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

MERGED="$ROOT/data/ml/chunk_qa_fixes_${MODE}.jsonl"
"$PY" - <<PY
import json
from pathlib import Path
root = Path("$ROOT")
mode = "$MODE"
by_id = {}
for fp in sorted(root.glob(f"data/ml/chunk_qa_fixes_{mode}_shard*.jsonl")):
    for line in fp.open(encoding="utf-8"):
        r = json.loads(line)
        cid = str(r.get("chunk_id") or "")
        if cid:
            by_id[cid] = r
out = root / f"data/ml/chunk_qa_fixes_{mode}.jsonl"
out.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print(json.dumps({"merged": len(by_id), "out": str(out)}, ensure_ascii=False))
PY

[ "$FAIL" -eq 0 ] || exit 1
echo "Done: $MERGED"
