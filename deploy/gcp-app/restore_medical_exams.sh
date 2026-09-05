#!/usr/bin/env bash
# Восстановление клинических данных МО из логического бэкапа.
#
# По умолчанию работает в режиме учения (--drill): распаковывает в отдельный
# каталог, сверяет контрольные суммы, проверяет целостность баз и печатает
# число записей. Прод не трогает. Именно в этом режиме скрипт должен
# прогоняться регулярно - иначе про бэкап неизвестно, восстанавливается ли он.
#
# Запись в прод (--target) требует ручной остановки приложения: подменять базы
# под работающим сервисом нельзя.
set -euo pipefail

BUCKET="${BACKUP_BUCKET:-gs://protocol-home-e1-backups}"
MODE="drill"
TARGET=""
ARCHIVE_URI=""

usage() {
  cat <<'USAGE'
Использование:
  restore_medical_exams.sh [--latest | --archive gs://...] [--drill | --target DIR]

  --latest              взять самый свежий архив из бакета (по умолчанию)
  --archive gs://...    конкретный архив
  --drill               учение: распаковать в /var/tmp и проверить (по умолчанию)
  --target DIR          восстановить в DIR; приложение должно быть остановлено

Примеры:
  # регулярное учение
  sudo restore_medical_exams.sh --latest --drill

  # настоящее восстановление
  sudo docker stop protocol-web
  sudo restore_medical_exams.sh --latest --target /var/data/medical_exams
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest) ARCHIVE_URI=""; shift ;;
    --archive) ARCHIVE_URI="$2"; shift 2 ;;
    --drill) MODE="drill"; shift ;;
    --target) MODE="restore"; TARGET="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "неизвестный аргумент: $1" >&2; usage; exit 2 ;;
  esac
done

log() { printf '[restore] %s\n' "$*"; }

# Итог учения тоже уходит в Cloud Logging: провал учения означает, что бэкапы
# есть, но развернуть их нельзя - знать об этом надо до аварии, а не во время.
report() {
  [[ "$MODE" == "drill" ]] || return 0
  local status="$1" detail="$2"
  gcloud logging write protocol-restore-drill \
    "{\"event\":\"restore_drill_finished\",\"status\":\"$status\",\"detail\":\"${detail//\"/}\"}" \
    --payload-type=json --severity="$([[ $status == ok ]] && echo INFO || echo ERROR)" \
    >/dev/null 2>&1 || printf '[restore] предупреждение: не удалось записать итог в Cloud Logging\n' >&2
}
die() {
  printf '[restore] ОШИБКА: %s\n' "$*" >&2
  report failed "$*"
  exit 1
}

if [[ -z "$ARCHIVE_URI" ]]; then
  log "ищу самый свежий архив в $BUCKET"
  ARCHIVE_URI="$(gcloud storage ls "$BUCKET/medical_exams/**/*.tar.zst" 2>/dev/null | sort | tail -1)"
  [[ -n "$ARCHIVE_URI" ]] || die "в бакете нет архивов"
fi
log "архив: $ARCHIVE_URI"

WORK="$(mktemp -d /var/tmp/protocol-restore.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

log "распаковываю"
gcloud storage cat "$ARCHIVE_URI" | zstd -d -q -c | tar -C "$WORK" -xf - \
  || die "архив не распаковался"

MANIFEST="$WORK/BACKUP_MANIFEST.txt"
[[ -f "$MANIFEST" ]] || die "в архиве нет BACKUP_MANIFEST.txt"

log "манифест:"
sed -n '1,6p' "$MANIFEST" | sed 's/^/    /'

# --- сверка контрольных сумм ---------------------------------------------------
# Манифест содержит sha256 всех файлов на момент снятия. Если байт побился в
# GCS или при распаковке, узнаём здесь, а не при попытке открыть базу.
log "сверяю sha256 всех файлов"
sums_bad=0
(cd "$WORK" && sed -n '/^--- sha256 ---$/,$p' BACKUP_MANIFEST.txt | tail -n +2 \
  | grep -v ' \./BACKUP_MANIFEST.txt$' > /tmp/_sums.$$ \
  && sha256sum -c --quiet /tmp/_sums.$$ 2>&1; rm -f /tmp/_sums.$$) || sums_bad=1
[[ "$sums_bad" == "0" ]] || die "контрольные суммы не сошлись - архив повреждён"
log "контрольные суммы сошлись"

# --- проверка баз --------------------------------------------------------------
log "проверяю базы"
for db in "$WORK"/warehouse/*.sqlite; do
  [[ -f "$db" ]] || continue
  python3 - "$db" <<'PY' || die "база $(basename "$db") не прошла проверку"
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path)
try:
    verdict = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if verdict != "ok":
        sys.exit(f"integrity_check: {verdict}")
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    total = 0
    for table in tables:
        total += conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
finally:
    conn.close()

name = path.rsplit("/", 1)[-1]
print(f"    {name}: таблиц {len(tables)}, записей {total}")
PY
done

jsonl_count="$(find "$WORK" -name '*.jsonl' | wc -l | tr -d ' ')"
log "jsonl файлов восстановлено: $jsonl_count"

expected_jsonl="$(sed -n 's/^jsonl_files=//p' "$MANIFEST")"
[[ "$jsonl_count" == "$expected_jsonl" ]] \
  || die "jsonl: восстановлено $jsonl_count, в манифесте $expected_jsonl"

if [[ "$MODE" == "drill" ]]; then
  log "учение прошло успешно, прод не тронут"
  report ok "архив $ARCHIVE_URI, jsonl $jsonl_count"
  exit 0
fi

# --- настоящее восстановление --------------------------------------------------
[[ -n "$TARGET" ]] || die "не задан --target"
if pgrep -f 'uvicorn|rag_server' >/dev/null 2>&1 || \
   sudo docker ps --format '{{.Names}}' 2>/dev/null | grep -q protocol-web; then
  die "приложение работает - останови его перед восстановлением (sudo docker stop protocol-web)"
fi

if [[ -d "$TARGET" ]]; then
  aside="$TARGET.before-restore.$(date -u +%Y%m%dT%H%M%SZ)"
  log "отодвигаю текущие данные в $aside"
  mv "$TARGET" "$aside" || die "не удалось отодвинуть $TARGET"
fi

mkdir -p "$TARGET"
cp -a "$WORK/." "$TARGET/"
rm -f "$TARGET/BACKUP_MANIFEST.txt"
log "восстановлено в $TARGET"
log "запусти приложение и проверь /health/live и одну страницу разбора"
