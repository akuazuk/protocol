#!/usr/bin/env bash
# Resolve Telegram chat_id after you send /start to your bot from @pavelopenai (or any account).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "Добавьте TELEGRAM_BOT_TOKEN в .env (бот от @BotFather)" >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import subprocess

token = os.environ["TELEGRAM_BOT_TOKEN"]
raw = subprocess.check_output(
    ["curl", "-sS", f"https://api.telegram.org/bot{token}/getUpdates"],
    text=True,
)
data = json.loads(raw)
for u in data.get("result", []):
    msg = u.get("message") or u.get("edited_message") or {}
    chat = msg.get("chat") or {}
    user = msg.get("from") or {}
    if chat.get("id"):
        print(
            f"chat_id={chat['id']}  "
            f"user=@{user.get('username', '?')}  "
            f"name={user.get('first_name', '')}"
        )
if not data.get("result"):
    print("Нет сообщений. Откройте @Cursor_Kravira_bot и нажмите /start, затем повторите.")
PY
