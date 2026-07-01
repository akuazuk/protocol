#!/usr/bin/env bash
# Фаза 2: переизвлечение слабых сводок (empty → priority no_exams) + аудит.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a && source .env && set +a
LOG=data/reports/phase2_reextract.log
mkdir -p data/reports
echo "=== Phase 2 start $(date -Iseconds) ===" | tee "$LOG"
python3 scripts/reextract_weak_summaries.py \
  --queue-file data/protocol_summaries/reextract_empty.json \
  --publish --resume --sleep 1 2>&1 | tee -a "$LOG"
python3 scripts/reextract_weak_summaries.py \
  --queue-file data/protocol_summaries/reextract_no_exams_prio.json \
  --publish --resume --sleep 1 2>&1 | tee -a "$LOG"
if [ -f scripts/export_protocol_summary_rules.py ]; then
  python3 scripts/export_protocol_summary_rules.py 2>&1 | tee -a "$LOG" || true
fi
python3 scripts/audit_summary_excerpts.py --json data/reports/summary_excerpt_audit_post_phase2.json 2>&1 | tee -a "$LOG"
echo "=== Phase 2 done $(date -Iseconds) ===" | tee -a "$LOG"
