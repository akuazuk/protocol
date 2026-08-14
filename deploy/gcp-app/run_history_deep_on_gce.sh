#!/usr/bin/env bash
# Слой B (+ опционально C) внутри protocol-web. Сильная модель только с --llm.
set -euo pipefail
DAY="${1:-yesterday}"
LIMIT="${2:-20}"
LLM_FLAG="${3:-}"
exec sudo docker exec \
  -e MO_LLM_EXECUTION_HOST=gce \
  -e RUN_HOST=gcp \
  -e MO_DATA_ROOT=/var/data/medical_exams \
  -e MO_WAREHOUSE=/var/data/medical_exams/warehouse/mo_analytics.sqlite \
  protocol-web \
  python scripts/run_mo_history_deep.py --date "$DAY" --limit "$LIMIT" ${LLM_FLAG}
