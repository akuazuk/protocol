#!/usr/bin/env bash
# Тестовые выгрузки mis_protocol на Render persistent disk (/var/data/mis_protocol).
# Данные с ПДн - не в git; только на диске Render / локально.
#
# Примеры:
#   bash scripts/render_mis_protocol_data.sh upload 2026-07
#   bash scripts/render_mis_protocol_data.sh list
#   bash scripts/render_mis_protocol_data.sh delete 2026-07
#   bash scripts/render_mis_protocol_data.sh delete-all
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="${ROOT}/data/mis_protocol"
SSH_HOST="${RENDER_SSH_HOST:-srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com}"
REMOTE_DIR="${RENDER_MIS_PROTOCOL_DIR:-/var/data/mis_protocol}"
SSH_OPTS=(-o ConnectTimeout=25 -o ServerAliveInterval=30)

usage() {
  cat <<EOF
Usage: $0 <upload|list|delete|delete-all> [YYYY-MM]

  upload YYYY-MM   scp parquet+csv+meta на Render (${REMOTE_DIR})
  list             показать файлы на Render
  delete YYYY-MM   удалить mis_protocol_YYYY-MM.* на Render
  delete-all       удалить весь каталог ${REMOTE_DIR} на Render
EOF
}

ssh_run() {
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"
}

cmd="${1:-}"
month="${2:-}"

case "$cmd" in
  upload)
    if [[ -z "$month" ]]; then
      echo "Нужен месяц: upload YYYY-MM" >&2
      exit 1
    fi
    base="mis_protocol_${month}"
    for ext in parquet csv meta.json; do
      f="${LOCAL_DIR}/${base}.${ext}"
      if [[ ! -f "$f" ]]; then
        echo "Нет файла: $f" >&2
        echo "Сначала: python3 scripts/export_mis_protocol_month.py --month ${month}" >&2
        exit 1
      fi
    done
    echo "Создаю ${REMOTE_DIR} на Render…"
    ssh_run "mkdir -p '${REMOTE_DIR}' && chmod 775 '${REMOTE_DIR}'"
    echo "Загружаю ${base}.{parquet,csv,meta.json} → ${SSH_HOST}:${REMOTE_DIR}/"
    scp "${SSH_OPTS[@]}" \
      "${LOCAL_DIR}/${base}.parquet" \
      "${LOCAL_DIR}/${base}.csv" \
      "${LOCAL_DIR}/${base}.meta.json" \
      "${SSH_HOST}:${REMOTE_DIR}/"
    echo "Готово. Проверка:"
    ssh_run "ls -lh '${REMOTE_DIR}/${base}'.*"
    ;;
  list)
    ssh_run "mkdir -p '${REMOTE_DIR}'; echo '=== ${REMOTE_DIR} ==='; ls -lh '${REMOTE_DIR}' 2>/dev/null || echo '(пусто)'"
    ;;
  delete)
    if [[ -z "$month" ]]; then
      echo "Нужен месяц: delete YYYY-MM" >&2
      exit 1
    fi
    base="mis_protocol_${month}"
    echo "Удаляю на Render: ${REMOTE_DIR}/${base}.*"
    ssh_run "rm -fv '${REMOTE_DIR}/${base}.parquet' '${REMOTE_DIR}/${base}.csv' '${REMOTE_DIR}/${base}.meta.json'"
    echo "Осталось:"
    ssh_run "ls -lh '${REMOTE_DIR}' 2>/dev/null || echo '(пусто)'"
    ;;
  delete-all)
    echo "Удаляю весь каталог ${REMOTE_DIR} на Render…"
    ssh_run "rm -rf '${REMOTE_DIR}' && echo deleted"
    ;;
  *)
    usage
    exit 1
    ;;
esac
