#!/usr/bin/env bash
# Скачать JSONL feedback с Render/on-prem для локального export_training_feedback.py
#
# Usage:
#   export METHODIST_TOKEN='...'
#   ./scripts/pull_methodist_feedback.sh https://protocol-bimy.onrender.com
#   ./scripts/pull_methodist_feedback.sh https://protocol-bimy.onrender.com data/ml/feedback_render
#   ./scripts/pull_methodist_feedback.sh https://protocol-bimy.onrender.com data/ml/feedback_render 2026-06-13
#
# После sync:
#   ML_FEEDBACK_DIR=data/ml/feedback_render python3 scripts/export_training_feedback.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_URL="${1:?Usage: $0 BASE_URL [OUT_DIR] [SINCE]}"
OUT_DIR="${2:-$ROOT/data/ml/feedback_render}"
SINCE="${3:-}"
TOKEN="${METHODIST_TOKEN:-${METHODIST_PIN:-}}"

if [[ -z "$TOKEN" ]]; then
  echo "Задайте METHODIST_TOKEN (или METHODIST_PIN) в окружении." >&2
  exit 1
fi

URL="${BASE_URL%/}/api/ml/feedback/export"
if [[ -n "$SINCE" ]]; then
  URL="${URL}?since=${SINCE}"
fi

mkdir -p "$OUT_DIR"
TMP="$(mktemp "${TMPDIR:-/tmp}/ml_feedback.XXXXXX.tar.gz")"
trap 'rm -f "$TMP"' EXIT

echo "GET $URL -> $OUT_DIR"
HTTP_HEADERS="$(mktemp)"
curl -fsSL \
  -H "X-Methodist-Token: $TOKEN" \
  -D "$HTTP_HEADERS" \
  -o "$TMP" \
  "$URL"

EVENT_COUNT="$(grep -i '^x-feedback-event-count:' "$HTTP_HEADERS" | tail -1 | tr -d '\r' | awk '{print $2}')"
rm -f "$HTTP_HEADERS"

# Архив: feedback/_manifest.json + feedback/*.jsonl
tar xzf "$TMP" -C "$OUT_DIR" --strip-components=1 feedback

echo "OK: events=${EVENT_COUNT:-?} -> $OUT_DIR"
ls -la "$OUT_DIR"/*.jsonl 2>/dev/null || true
