#!/usr/bin/env bash
# Заливка rich_chunks.jsonl на Persistent Disk Render (SSH + rsync/scp).
#
# Предварительно в Render Dashboard:
#   - Disk mount: /var/data
#   - Env: RAG_CHUNKS_DIR=/var/data
#         (или RAG_CHUNKS_JSONL=/var/data/output/rich_chunks/rich_chunks.jsonl)
#   - SSH: Account Settings → SSH Public Keys
#
# Один раз ввести passphrase (macOS):
#   eval "$(ssh-agent -s)"
#   ssh-add --apple-use-keychain ~/.ssh/id_ed25519
#
# Usage:
#   ./scripts/upload_rich_chunks_render.sh srv-xxxxx@ssh.oregon.render.com --gzip
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_TARGET="${1:?Usage: $0 srv-xxx@ssh.REGION.render.com [--gzip]}"
USE_GZIP="${2:-}"

if [[ "$SSH_TARGET" == "ssh" ]]; then
  SSH_TARGET="${2:?Usage: $0 srv-xxx@ssh.REGION.render.com [--gzip]}"
  USE_GZIP="${3:-}"
fi

SRC_JSONL="$ROOT/output/rich_chunks/rich_chunks.v2.jsonl"
if [[ ! -f "$SRC_JSONL" ]]; then
  SRC_JSONL="$ROOT/output/rich_chunks/rich_chunks.jsonl"
fi
SRC_MANIFEST="$ROOT/output/rich_chunks/_manifest.json"
REMOTE_DIR="/var/data/output/rich_chunks"
REMOTE_JSONL="$REMOTE_DIR/rich_chunks.jsonl"

if [[ ! -f "$SRC_JSONL" ]]; then
  echo "Нет файла: $SRC_JSONL" >&2
  echo "Сначала: .venv/bin/python scripts/build_rich_chunks.py" >&2
  exit 1
fi

# Одно SSH-соединение на весь скрипт — passphrase спрашивается один раз
CTRL_SOCK="${TMPDIR:-/tmp}/render-upload-$$.sock"
cleanup() {
  ssh -S "$CTRL_SOCK" -O exit "$SSH_TARGET" 2>/dev/null || true
  rm -f "$CTRL_SOCK"
}
trap cleanup EXIT

SSH_OPTS=(
  -o ControlMaster=auto
  -o "ControlPath=$CTRL_SOCK"
  -o ControlPersist=300
  -o StrictHostKeyChecking=accept-new
  -o "IdentityFile=${RENDER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
  -o AddKeysToAgent=yes
  -o UseKeychain=yes
)

ssh_cmd() {
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"
}

rsync_cmd() {
  RSYNC_RSH="ssh ${SSH_OPTS[*]}" rsync -avP --progress "$@"
}

echo "Локальный файл: $SRC_JSONL ($(du -h "$SRC_JSONL" | awk '{print $1}'))"
echo "SSH: $SSH_TARGET"
echo "Удалённый путь: $REMOTE_JSONL"
echo ""
echo "Подключение (passphrase — один раз на весь скрипт)..."
ssh_cmd "mkdir -p '$REMOTE_DIR'"

if [[ "$USE_GZIP" == "--gzip" ]]; then
  TMP_GZ="$(mktemp "${TMPDIR:-/tmp}/rich_chunks.XXXXXX.jsonl.gz")"
  trap 'rm -f "$TMP_GZ"; cleanup' EXIT
  echo "Сжатие (gzip)..."
  gzip -c "$SRC_JSONL" > "$TMP_GZ"
  echo "Загрузка $(du -h "$TMP_GZ" | awk '{print $1}')..."
  rsync_cmd "$TMP_GZ" "$SSH_TARGET:$REMOTE_DIR/rich_chunks.jsonl.gz"
  echo "Распаковка на сервере (~1-2 мин)..."
  ssh_cmd "gunzip -cf '$REMOTE_DIR/rich_chunks.jsonl.gz' > '$REMOTE_JSONL.tmp' && mv '$REMOTE_JSONL.tmp' '$REMOTE_JSONL' && rm -f '$REMOTE_DIR/rich_chunks.jsonl.gz'"
else
  echo "Загрузка rsync..."
  rsync_cmd "$SRC_JSONL" "$SSH_TARGET:$REMOTE_JSONL"
fi

if [[ -f "$SRC_MANIFEST" ]]; then
  rsync_cmd "$SRC_MANIFEST" "$SSH_TARGET:$REMOTE_DIR/_manifest.json"
fi

echo ""
echo "Проверка на сервере:"
ssh_cmd "ls -lh '$REMOTE_JSONL' && wc -l < '$REMOTE_JSONL'"

echo ""
echo "Готово. Restart сервиса в Render, затем:"
echo "  curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool"
