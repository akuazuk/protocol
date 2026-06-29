#!/usr/bin/env python3
"""Telegram Bot API: уведомления о статусе pipeline (без секретов в git)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_load import load_project_env

load_project_env(ROOT)


def _valid_chat_id(raw: str) -> bool:
    s = (raw or "").strip()
    return s.lstrip("-").isdigit() and s not in ("...", "…")


def telegram_enabled() -> bool:
    if os.environ.get("TELEGRAM_NOTIFY_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() and _valid_chat_id(os.environ.get("TELEGRAM_CHAT_ID") or ""))


def send_telegram(text: str, *, parse_mode: str = "") -> bool:
    """Отправить сообщение. True если ушло или notify выключен без ошибки."""
    if not telegram_enabled():
        return False
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    chat_id = os.environ["TELEGRAM_CHAT_ID"].strip()
    body: dict[str, str] = {"chat_id": chat_id, "text": text[:4090]}
    if parse_mode:
        body["parse_mode"] = parse_mode
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return bool(payload.get("ok"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
        print(f"telegram send failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Send Telegram notification")
    parser.add_argument("message", nargs="?", default="", help="Text to send")
    parser.add_argument("--file", help="Read message from file")
    parser.add_argument("--check", action="store_true", help="Print enabled/disabled and exit 0/1")
    args = parser.parse_args()

    if args.check:
        ok = telegram_enabled()
        print("enabled" if ok else "disabled")
        return 0 if ok else 1

    text = args.message
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    if not text:
        print("empty message", file=sys.stderr)
        return 2
    return 0 if send_telegram(text) else 3


if __name__ == "__main__":
    raise SystemExit(main())
