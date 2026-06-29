#!/usr/bin/env python3
"""Показать chat_id после сообщения боту (/start).

  python3 scripts/telegram_get_chat_id.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_load import load_project_env

load_project_env(ROOT)


def main() -> int:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print("Задайте TELEGRAM_BOT_TOKEN в .env (от @BotFather)", file=sys.stderr)
        return 1
    url = f"https://api.telegram.org/bot{token}/getUpdates?limit=20"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        print("Telegram API error", data, file=sys.stderr)
        return 2
    seen: dict[int, tuple[str, str]] = {}
    for upd in data.get("result") or []:
        for key in ("message", "edited_message"):
            msg = upd.get(key)
            if not msg:
                continue
            chat = msg.get("chat") or {}
            cid = chat.get("id")
            if cid is None:
                continue
            seen[int(cid)] = (str(chat.get("first_name") or ""), str(chat.get("username") or ""))
        cb = upd.get("callback_query")
        if cb:
            chat = (cb.get("message") or {}).get("chat") or {}
            cid = chat.get("id")
            if cid is not None:
                seen[int(cid)] = (str(chat.get("first_name") or ""), str(chat.get("username") or ""))
    if not seen:
        print("Нет сообщений. Напишите боту /start в Telegram и запустите снова.")
        return 3
    print("Скопируйте в .env:")
    for cid in sorted(seen):
        name, uname = seen[cid]
        extra = f" ({uname})" if uname else ""
        print(f"TELEGRAM_CHAT_ID={cid}  # {name}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
