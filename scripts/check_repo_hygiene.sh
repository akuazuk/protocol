#!/usr/bin/env bash
# Lightweight repository hygiene audit (read-only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Repo: $ROOT"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo

echo "== Git status (short) =="
git status --short || true
echo

echo "== Potentially heavy local directories =="
for p in corpus_vector_index output/rich_chunks output/rich_meta data/ml/chunk_qa_cache data/ml/chunk_qa_shards; do
  if [[ -d "$p" ]]; then
    du -sh "$p" 2>/dev/null || true
  fi
done
echo

echo "== Local logs/state snapshots =="
rg -n ".*" data/ml/reports -g "*.log" -g "*.done" -g "*_state.json" --files-with-matches 2>/dev/null | sed -n '1,40p' || true
echo

echo "== Guard checks =="
scripts/git_safe_start.sh >/tmp/protocol_safe_start_hygiene.out 2>&1 || true
rg -n "Branch|HEAD|Divergence|ahead|behind|WARN|OK|Recommended" /tmp/protocol_safe_start_hygiene.out || true
echo

echo "Done. No files changed."
