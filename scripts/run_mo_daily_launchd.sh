#!/bin/bash
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-main}"
PYTHON_BIN="${MO_PYTHON:-$(command -v python3)}"

case "$MODE" in
  main)
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" --catch-up --reconcile-days 3
    ;;
  retry)
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" --catch-up
    ;;
  hourly)
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" --catch-up --catch-up-limit 31
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
