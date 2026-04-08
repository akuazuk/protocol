#!/usr/bin/env bash
# Полный локальный прогон: pytest + оценка поиска на мини-корпусе.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export RAG_GEMINI_EMBED_RERANK="${RAG_GEMINI_EMBED_RERANK:-0}"
python3 -m pytest tests/ -q
python3 eval/search_quality_eval.py --embed-off --mini --golden eval/golden_queries.jsonl
echo "OK: pytest + search_quality_eval (mini)"
