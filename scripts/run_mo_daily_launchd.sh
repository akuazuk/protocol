#!/bin/bash
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-main}"
PYTHON_BIN="${MO_PYTHON:-$(command -v python3)}"

case "$MODE" in
  main)
    # Основной приём ~06:00 Europe/Minsk: вчера + catch-up + reconcile 3 дней.
    # В понедельник дополнительно перезаписываем прошлую полную неделю Пн-Вс.
    EXTRA=()
    if [ "$(TZ=Europe/Minsk date +%u)" = "1" ]; then
      EXTRA+=(--previous-week)
    fi
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" \
      --catch-up --reconcile-days 3 "${EXTRA[@]}"
    ;;
  retry)
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" --catch-up
    ;;
  hourly)
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" \
      --catch-up --catch-up-limit 31
    ;;
  weekly)
    # Страховка понедельника: явная перезапись прошлой недели (если утренний main не успел).
    exec /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" --previous-week
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
