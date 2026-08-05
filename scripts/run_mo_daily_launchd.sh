#!/bin/bash
set -euo pipefail

ROOT="${PROTOCOL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODE="${1:-main}"
PYTHON_BIN="${MO_PYTHON:-$(command -v python3)}"
STATE_DIR="$ROOT/data/medical_exams/state"
STATE_FILE="$STATE_DIR/pipeline.json"
RUN_LOCK="$STATE_DIR/launchd-run.lock"
PIPELINE_LOCK="$STATE_DIR/pipeline.lock"

# Секреты не кладём в plist: подхватываем из .env проекта (и PROTOCOL_ENV_FILE).
load_dotenv_keys() {
  local env_file="${PROTOCOL_ENV_FILE:-$ROOT/.env}"
  local key value line
  if [ ! -f "$env_file" ]; then
    return 0
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      METHODIST_TOKEN|METHODIST_PIN|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|TELEGRAM_NOTIFY)
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        if [ -z "${!key:-}" ]; then
          export "$key=$value"
        fi
        ;;
    esac
  done < "$env_file"
}
load_dotenv_keys

# The pipeline lock protects scoring only. Keep publication under the same
# launchd-level lock so retry/hourly jobs cannot VACUUM and upload the warehouse
# concurrently.
mkdir -p "$STATE_DIR"
clear_stale_pid_lock() {
  local lock_path="$1"
  local owner=""
  if [ ! -f "$lock_path" ]; then
    return 0
  fi
  owner="$(tr -d '[:space:]' < "$lock_path" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    return 0
  fi
  rm -f "$lock_path"
  echo "снят stale lock $lock_path (pid=${owner:-unknown})"
}
acquire_run_lock() {
  clear_stale_pid_lock "$RUN_LOCK"
  clear_stale_pid_lock "$PIPELINE_LOCK"
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
  if [ -z "${METHODIST_TOKEN:-${METHODIST_PIN:-}}" ]; then
    echo "METHODIST_TOKEN не задан: freshness-check после publish вернёт 403" >&2
  fi
  "$PYTHON_BIN" "$ROOT/scripts/publish_mo_to_render.py" \
    --methodist-token "${METHODIST_TOKEN:-${METHODIST_PIN:-}}" || {
    local rc=$?
    echo "публикация в прод не удалась (код $rc): данные на месте, повторить вручную" >&2
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
      "$PYTHON_BIN" "$ROOT/scripts/telegram_notify.py" \
        "МО publish fail mode=$MODE rc=$rc host=$(hostname -s)" >/dev/null 2>&1 || true
    fi
    return "$rc"
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

# Gemini с Mac часто geo-block - night LLM и action-judge гоняем на Render после publish.
run_render_llm_for_yesterday() {
  local day
  day="$(TZ=Europe/Minsk date -v-1d +%Y-%m-%d 2>/dev/null || TZ=Europe/Minsk date -d yesterday +%Y-%m-%d)"
  if [ "${MO_RENDER_LLM_AFTER_PUBLISH:-1}" = "0" ]; then
    echo "МО: remote LLM после publish отключён (MO_RENDER_LLM_AFTER_PUBLISH=0)"
    return 0
  fi
  if [ ! -x "$ROOT/scripts/run_mo_render_llm_backfill.sh" ]; then
    chmod +x "$ROOT/scripts/run_mo_render_llm_backfill.sh" 2>/dev/null || true
  fi
  echo "МО: remote LLM backfill на Render за $day"
  bash "$ROOT/scripts/run_mo_render_llm_backfill.sh" "$day" "$day" \
    || echo "МО: remote LLM backfill завершился с ошибкой (день уже опубликован)" >&2
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
    if [ "$PIPELINE_CHANGED" = "1" ]; then
      run_render_llm_for_yesterday
    fi
    ;;
  retry)
    if catch_up_needed; then
      run_pipeline --catch-up
      publish_if_changed
      if [ "$PIPELINE_CHANGED" = "1" ]; then
        run_render_llm_for_yesterday
      fi
    else
      echo "МО: catch-up не требуется, retry пропущен"
    fi
    ;;
  hourly)
    if catch_up_needed; then
      run_pipeline --catch-up --catch-up-limit 31
      publish_if_changed
      if [ "$PIPELINE_CHANGED" = "1" ]; then
        run_render_llm_for_yesterday
      fi
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
  llm-yesterday)
    # Ручной/страховочный прогон LLM на Render за вчера.
    run_render_llm_for_yesterday
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

exit "$PIPELINE_STATUS"
