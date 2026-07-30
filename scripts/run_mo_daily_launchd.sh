#!/bin/bash
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-main}"
PYTHON_BIN="${MO_PYTHON:-$(command -v python3)}"

# Публикация на диск Render после успешного прогона: без неё прод показывает
# вчерашние данные только на этой машине. MO_PUBLISH_TO_RENDER=0 отключает.
publish_to_render() {
  if [ "${MO_PUBLISH_TO_RENDER:-1}" != "1" ]; then
    return 0
  fi
  "$PYTHON_BIN" "$ROOT/scripts/publish_mo_to_render.py" \
    --methodist-token "${METHODIST_TOKEN:-}" || {
    echo "публикация в прод не удалась: данные на месте, повторить вручную" >&2
    return 1
  }
}

# Прогон не должен обрывать публикацию: неполный день тоже стоит показать в проде,
# а код возврата конвейера отдаём launchd в конце.
PIPELINE_STATUS=0
run_pipeline() {
  /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" "$@" \
    || PIPELINE_STATUS=$?
}

case "$MODE" in
  main)
    # Основной приём ~06:00 Europe/Minsk: вчера + catch-up + reconcile 3 дней.
    # В понедельник дополнительно перезаписываем прошлую полную неделю Пн-Вс.
    MAIN_ARGS=(--catch-up --reconcile-days 3)
    if [ "$(TZ=Europe/Minsk date +%u)" = "1" ]; then
      MAIN_ARGS+=(--previous-week)
    fi
    run_pipeline "${MAIN_ARGS[@]}"
    publish_to_render
    ;;
  retry)
    run_pipeline --catch-up
    publish_to_render
    ;;
  hourly)
    run_pipeline --catch-up --catch-up-limit 31
    publish_to_render
    ;;
  weekly)
    # Страховка понедельника: явная перезапись прошлой недели (если утренний main не успел).
    run_pipeline --previous-week
    publish_to_render
    ;;
  publish)
    # Только публикация: пригодится после ручного пересчёта истории.
    publish_to_render
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

exit "$PIPELINE_STATUS"
