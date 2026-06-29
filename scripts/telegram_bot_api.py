#!/usr/bin/env python3
"""Минимальный клиент Telegram Bot API (без внешних зависимостей)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _token() -> str:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def _chat_id() -> str:
    return (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()


def _api(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = _token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not out.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {out}")
    return out


def send_message(
    text: str,
    *,
    chat_id: str | None = None,
    reply_markup: dict[str, Any] | None = None,
    parse_mode: str = "",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "chat_id": chat_id or _chat_id(),
        "text": text[:4090],
    }
    if parse_mode:
        body["parse_mode"] = parse_mode
    if reply_markup:
        body["reply_markup"] = reply_markup
    return _api("sendMessage", body)


def answer_callback(callback_query_id: str, text: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        body["text"] = text[:190]
    return _api("answerCallbackQuery", body)


def get_updates(offset: int = 0, timeout: int = 25) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
    if offset:
        body["offset"] = offset
    try:
        out = _api("getUpdates", body)
    except (urllib.error.URLError, TimeoutError):
        return []
    return list(out.get("result") or [])


def inline_keyboard(rows: list[list[tuple[str, str]]]) -> dict[str, Any]:
    """rows: [[(label, callback_data), ...], ...]"""
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data[:64]} for label, data in row]
            for row in rows
        ]
    }
