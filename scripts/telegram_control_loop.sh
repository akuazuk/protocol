#!/usr/bin/env bash
# Слушает Telegram: кнопки Да/Нет и текстовые ответы → git push, smoke, embed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "${PY:-$ROOT/.venv/bin/python}" scripts/telegram_control.py loop "$@"
