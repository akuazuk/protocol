#!/bin/bash
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-main}"
PYTHON_BIN="${MO_PYTHON:-$(command -v python3)}"
STATE_DIR="$ROOT/data/medical_exams/state"
STATE_FILE="$STATE_DIR/pipeline.json"
RUN_LOCK="$STATE_DIR/launchd-run.lock"

# The pipeline lock protects scoring only. Keep publication under the same
# launchd-level lock so retry/hourly jobs cannot VACUUM and upload the warehouse
# concurrently.
mkdir -p "$STATE_DIR"
acquire_run_lock() {
  if (set -o noclobber; printf '%s\n' "$$" > "$RUN_LOCK") 2>/dev/null; then
    return 0
  fi
  local owner=""
  owner="$(cat "$RUN_LOCK" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    echo "МО launchd уже выполняется (pid $owner), режим $MODE пропущен"
    exit 0
  fi
  rm -f "$RUN_LOCK"
  (set -o noclobber; printf '%s\n' "$$" > "$RUN_LOCK") 2>/dev/null || exit 0
}
release_run_lock() {
  if [ "$(cat "$RUN_LOCK" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$RUN_LOCK"
  fi
}
acquire_run_lock
trap release_run_lock EXIT INT TERM

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
PIPELINE_CHANGED=0
state_fingerprint() {
  if [ -f "$STATE_FILE" ]; then
    shasum -a 256 "$STATE_FILE" | awk '{print $1}'
  else
    printf 'missing\n'
  fi
}
catch_up_needed() {
  if [ ! -s "$STATE_FILE" ]; then
    return 0
  fi
  local yesterday
  yesterday="$(TZ=Europe/Minsk date -v-1d +%F)"
  # Mirrors the cheap part of PipelineState selection. Avoid importing the
  # scoring stack every hour when yesterday is already settled.
  jq -e --arg day "$yesterday" '
    (.dates[$day].status != "success")
    or any(.dates[]; .status == "partial" and ((.attempts // 0) < 4))
  ' "$STATE_FILE" >/dev/null 2>&1
}
run_pipeline() {
  local before after
  before="$(state_fingerprint)"
  /usr/bin/caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_mo_daily_report.py" "$@" \
    || PIPELINE_STATUS=$?
  after="$(state_fingerprint)"
  if [ "$before" != "$after" ]; then
    PIPELINE_CHANGED=1
  fi
}
publish_if_changed() {
  if [ "$PIPELINE_CHANGED" = "1" ]; then
    publish_to_render
  else
    echo "МО: изменений состояния нет, публикация пропущена"
  fi
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
    publish_if_changed
    ;;
  retry)
    if catch_up_needed; then
      run_pipeline --catch-up
      publish_if_changed
    else
      echo "МО: catch-up не требуется, retry пропущен"
    fi
    ;;
  hourly)
    if catch_up_needed; then
      run_pipeline --catch-up --catch-up-limit 31
      publish_if_changed
    else
      echo "МО: catch-up не требуется, hourly пропущен"
    fi
    ;;
  weekly)
    # Страховка понедельника: явная перезапись прошлой недели (если утренний main не успел).
    run_pipeline --previous-week
    publish_if_changed
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
