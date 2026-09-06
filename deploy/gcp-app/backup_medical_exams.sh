#!/usr/bin/env bash
# Логический бэкап клинических данных МО в GCS.
#
# Снапшоты диска (resource policy protocol-daily-snapshots) восстанавливают весь
# диск целиком и только в тот же проект GCP. Этот скрипт делает второй,
# независимый контур: переносимый архив только незаменимой части /var/data.
#
# Что бэкапим и почему именно это:
#   medical_exams/  - оценки, разборы, обратная связь врачей. Заново не собрать.
#   rceth/          - НЕТ, 12 ГБ, пересобирается из внешнего реестра.
#   protocol_corpus/- НЕТ, пересобирается из PDF в репозитории.
#
# Базы SQLite копируются через sqlite3 backup API, а не cp: на живой базе в
# режиме WAL копия файла может застать транзакцию посередине. backup API даёт
# согласованный снимок без остановки приложения.
#
# Запуск: systemd timer protocol-backup.timer (см. install_backup_timer.sh).
set -euo pipefail

BUCKET="${BACKUP_BUCKET:-gs://protocol-home-e1-backups}"
DATA_DIR="${BACKUP_DATA_DIR:-/var/data/medical_exams}"
WORK_DIR="${BACKUP_WORK_DIR:-/var/tmp/protocol-backup}"
KEEP_LOCAL="${BACKUP_KEEP_LOCAL:-0}"

log() { printf '[backup] %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# Итог уходит в Cloud Logging: на нём висят два алерта - на явный сбой и на
# отсутствие успеха за 48 часов. Второй важнее: он ловит случай, когда таймер
# вообще не сработал и сбоя поэтому нет.
report() {
  local status="$1" detail="$2"
  gcloud logging write protocol-backup \
    "{\"event\":\"backup_finished\",\"status\":\"$status\",\"detail\":\"${detail//\"/}\"}" \
    --payload-type=json --severity="$([[ $status == ok ]] && echo INFO || echo ERROR)" \
    >/dev/null 2>&1 || printf '[backup] предупреждение: не удалось записать итог в Cloud Logging\n' >&2
}
die() {
  printf '[backup] ОШИБКА: %s\n' "$*" >&2
  report failed "$*"
  exit 1
}

[[ -d "$DATA_DIR" ]] || die "нет каталога данных $DATA_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGE="$WORK_DIR/$STAMP"
ARCHIVE="$WORK_DIR/medical_exams-$STAMP.tar.zst"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/warehouse"

# --- 1. Согласованные копии баз ------------------------------------------------
db_count=0
while IFS= read -r db; do
  name="$(basename "$db")"
  log "база $name"
  python3 - "$db" "$STAGE/warehouse/$name" <<'PY' || die "не удалось скопировать $name"
import sqlite3
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
# immutable=0 + режим только на чтение: не мешаем писателям приложения.
src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=60)
dst = sqlite3.connect(dst_path)
try:
    src.backup(dst)
finally:
    dst.close()
    src.close()

check = sqlite3.connect(dst_path)
try:
    verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
finally:
    check.close()
if verdict != "ok":
    sys.exit(f"integrity_check копии вернул {verdict!r}")
PY
  db_count=$((db_count + 1))
done < <(find "$DATA_DIR" -name '*.sqlite' -not -name '*-wal' -not -name '*-shm' | sort)

log "баз скопировано: $db_count"
[[ "$db_count" -gt 0 ]] || die "не найдено ни одной базы - проверь $DATA_DIR"

# --- 2. JSONL и остальные файлы данных ----------------------------------------
# Исключаем -wal/-shm: они относятся к оригиналам баз, для копий бесполезны.
rsync -a \
  --exclude 'warehouse/*.sqlite' \
  --exclude 'warehouse/*.sqlite-wal' \
  --exclude 'warehouse/*.sqlite-shm' \
  --exclude 'logs/' \
  --exclude '*.tmp' \
  "$DATA_DIR/" "$STAGE/" || die "rsync данных не прошёл"

jsonl_count="$(find "$STAGE" -name '*.jsonl' | wc -l | tr -d ' ')"
log "jsonl файлов: $jsonl_count"

# --- 3. Манифест: что внутри, чтобы восстановление можно было сверить ----------
{
  echo "stamp=$STAMP"
  echo "source_host=$(hostname)"
  echo "source_dir=$DATA_DIR"
  echo "databases=$db_count"
  echo "jsonl_files=$jsonl_count"
  echo "stage_bytes=$(du -sb "$STAGE" | cut -f1)"
  echo "--- sha256 ---"
  (cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
} > "$STAGE/BACKUP_MANIFEST.txt"

# --- 4. Архив и выгрузка -------------------------------------------------------
tar -C "$STAGE" -c . | zstd -3 -q -o "$ARCHIVE" || die "не удалось собрать архив"
size_mb=$(( $(stat -c %s "$ARCHIVE") / 1048576 ))
log "архив $size_mb МБ"

target="$BUCKET/medical_exams/$(date -u +%Y/%m)/$(basename "$ARCHIVE")"
gcloud storage cp "$ARCHIVE" "$target" --quiet || die "выгрузка в $target не прошла"

# --- 5. Проверка, что выгруженное читается ------------------------------------
# Без этого шага скрипт рапортует успех даже если в бакете лежит обрезанный файл.
remote_size="$(gcloud storage ls --long "$target" 2>/dev/null | awk 'NR==1{print $1}')"
local_size="$(stat -c %s "$ARCHIVE")"
[[ "$remote_size" == "$local_size" ]] || die "размер в бакете $remote_size != локального $local_size"

gcloud storage cat "$target" 2>/dev/null | zstd -d -q -c | tar -tf - >/dev/null \
  || die "архив в бакете не распаковывается"

log "готово: $target ($size_mb МБ, баз $db_count, jsonl $jsonl_count)"
report ok "$size_mb МБ, баз $db_count, jsonl $jsonl_count"
[[ "$KEEP_LOCAL" == "1" ]] || rm -f "$ARCHIVE"
