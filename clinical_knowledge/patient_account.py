"""Локальный аккаунт пациента (stub для облачной синхронизации)."""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

# In-memory store for dev; prod → Postgres + encryption
_SESSIONS: dict[str, dict[str, Any]] = {}


def create_guest_session(*, device_hint: str | None = None) -> dict[str, Any]:
    token = secrets.token_urlsafe(24)
    sid = hashlib.sha256(token.encode()).hexdigest()[:16]
    _SESSIONS[sid] = {
        "created_at": int(time.time()),
        "device_hint": (device_hint or "")[:64],
        "history": [],
    }
    return {"ok": True, "session_token": token, "session_id": sid}


def sync_history(session_token: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    sid = _session_id_from_token(session_token)
    if not sid or sid not in _SESSIONS:
        return {"ok": False, "error": "invalid_session"}
    safe: list[dict[str, Any]] = []
    for e in entries[:10]:
        if not isinstance(e, dict):
            continue
        safe.append(
            {
                "ts": str(e.get("ts", ""))[:32],
                "pct": e.get("pct"),
                "light": str(e.get("light", ""))[:12],
                "label": str(e.get("label", ""))[:120],
                "summary": str(e.get("summary", ""))[:240],
            }
        )
    _SESSIONS[sid]["history"] = safe
    _SESSIONS[sid]["updated_at"] = int(time.time())
    return {"ok": True, "stored": len(safe)}


def get_history(session_token: str) -> dict[str, Any]:
    sid = _session_id_from_token(session_token)
    if not sid or sid not in _SESSIONS:
        return {"ok": False, "error": "invalid_session"}
    return {"ok": True, "history": list(_SESSIONS[sid].get("history") or [])}


def _session_id_from_token(token: str) -> str | None:
    raw = (token or "").strip()
    if not raw:
        return None
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
