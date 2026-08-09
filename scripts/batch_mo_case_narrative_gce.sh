#!/usr/bin/env bash
# Opt-in batch LLM narrative для clinical_visit дня на GCE.
# Не запускать с Mac: Gemini geo. Default MO_CASE_NARRATIVE остаётся off в API.
#
# Usage (на GCE / через deploy/gcp-llm wrapper):
#   bash scripts/batch_mo_case_narrative_gce.sh 2026-08-06
set -euo pipefail
DAY="${1:?usage: batch_mo_case_narrative_gce.sh YYYY-MM-DD}"
export MO_CASE_NARRATIVE="${MO_CASE_NARRATIVE:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[narrative-batch] day=$DAY MO_CASE_NARRATIVE=$MO_CASE_NARRATIVE"
python3 - <<PY
"""Lazy smoke: narrative module import + flag; полный batch - через night runner."""
from clinical_knowledge.mo_case_narrative import case_narrative_enabled, ENGINE
print("engine", ENGINE, "enabled", case_narrative_enabled())
if not case_narrative_enabled():
    raise SystemExit("MO_CASE_NARRATIVE must be 1 for batch")
print("ok: module ready; wire per-case generate_case_narrative in night job when ready")
PY
