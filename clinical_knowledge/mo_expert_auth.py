"""Авторизация врача-эксперта: логин/пароль + session в SQLite warehouse.

См. docs/plans/2026-08-05-mo-expert-reviewer-portal-v1.md.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from .mo_backend import _connect, _utc

EXPERT_ROLE = "expert"
SESSION_TTL_HOURS = 12
REPORTS_MIN_DATE_DEFAULT = "2026-08-01"
SESSION_HEADER = "x-expert-session"

EXPERT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_expert_user (
  expert_id TEXT PRIMARY KEY,
  login TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  display_name TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_expert_session (
  session_id TEXT PRIMARY KEY,
  expert_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_crm_expert_session_expert
  ON crm_expert_session(expert_id, expires_at);
"""

# Префиксы API, доступные роли expert (после логина).
EXPERT_ALLOWED_PREFIXES = (
    "/api/expert/",
    "/api/methodist/mo/capabilities",
    "/api/methodist/mo/daily-report",
    "/api/methodist/mo/reports",
    "/api/methodist/mo/cases",
    "/api/methodist/mo/review-packs",
    "/api/methodist/mo/rubric-summary",
    "/api/methodist/mo/freshness",
    "/api/methodist/mo/health",
    "/api/methodist/mo/meta",
)


def ensure_expert_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    db = conn or _connect()
    try:
        db.executescript(EXPERT_SCHEMA_SQL)
        if own:
            db.commit()
    finally:
        if own:
            db.close()


def reports_min_date() -> str:
    raw = (os.environ.get("MO_EXPERT_REPORTS_MIN_DATE") or REPORTS_MIN_DATE_DEFAULT).strip()
    return raw[:10] if len(raw) >= 10 else REPORTS_MIN_DATE_DEFAULT


def _hash_password(password: str, *, salt: str | None = None) -> str:
    salt_hex = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    ).hex()
    return f"scrypt${salt_hex}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest = stored.split("$", 2)
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    candidate = _hash_password(password, salt=salt_hex)
    return hmac.compare_digest(candidate, stored)


def upsert_expert_user(
    *,
    login: str,
    password: str,
    display_name: str = "",
    active: bool = True,
) -> dict[str, Any]:
    login_norm = str(login or "").strip().lower()
    if not login_norm or len(login_norm) < 2:
        raise ValueError("login_required")
    if len(password) < 8:
        raise ValueError("password_too_short")
    ensure_expert_schema()
    now = _utc()
    expert_id = str(uuid.uuid4())
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        existing = conn.execute(
            "SELECT expert_id FROM crm_expert_user WHERE login=?",
            (login_norm,),
        ).fetchone()
        if existing:
            expert_id = str(existing["expert_id"])
            conn.execute(
                """UPDATE crm_expert_user
                   SET password_hash=?, display_name=?, active=?, created_at=COALESCE(created_at, ?)
                   WHERE expert_id=?""",
                (
                    _hash_password(password),
                    (display_name or login_norm)[:120],
                    1 if active else 0,
                    now,
                    expert_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO crm_expert_user(
                     expert_id, login, password_hash, display_name, active, created_at
                   ) VALUES (?,?,?,?,?,?)""",
                (
                    expert_id,
                    login_norm,
                    _hash_password(password),
                    (display_name or login_norm)[:120],
                    1 if active else 0,
                    now,
                ),
            )
        conn.commit()
    return {"ok": True, "expert_id": expert_id, "login": login_norm, "active": active}


def ensure_bootstrap_expert() -> dict[str, Any] | None:
    """Создать одного эксперта из env, если таблицы пусты."""
    login = (os.environ.get("MO_EXPERT_BOOTSTRAP_LOGIN") or "expert").strip().lower()
    password = (os.environ.get("MO_EXPERT_BOOTSTRAP_PASSWORD") or "").strip()
    if not password:
        return None
    ensure_expert_schema()
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        count = conn.execute("SELECT COUNT(*) AS n FROM crm_expert_user").fetchone()["n"]
        if int(count or 0) > 0:
            return {"ok": True, "bootstrapped": False, "existing": int(count)}
    created = upsert_expert_user(
        login=login,
        password=password,
        display_name=(os.environ.get("MO_EXPERT_BOOTSTRAP_NAME") or "Врач-эксперт").strip(),
    )
    created["bootstrapped"] = True
    return created


def login_expert(*, login: str, password: str) -> dict[str, Any]:
    ensure_bootstrap_expert()
    login_norm = str(login or "").strip().lower()
    if not login_norm or not password:
        raise PermissionError("invalid_credentials")
    ensure_expert_schema()
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        row = conn.execute(
            "SELECT * FROM crm_expert_user WHERE login=? AND active=1",
            (login_norm,),
        ).fetchone()
        if not row or not _verify_password(password, str(row["password_hash"])):
            raise PermissionError("invalid_credentials")
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=SESSION_TTL_HOURS)
        conn.execute(
            """INSERT INTO crm_expert_session(
                 session_id, expert_id, created_at, expires_at, last_seen_at
               ) VALUES (?,?,?,?,?)""",
            (
                session_id,
                row["expert_id"],
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            ),
        )
        conn.commit()
    return {
        "ok": True,
        "role": EXPERT_ROLE,
        "session_token": session_id,
        "login": login_norm,
        "display_name": str(row["display_name"] or login_norm),
        "reports_min_date": reports_min_date(),
        "expires_in_hours": SESSION_TTL_HOURS,
    }


def logout_expert(session_token: str) -> dict[str, Any]:
    token = str(session_token or "").strip()
    if not token:
        return {"ok": True}
    ensure_expert_schema()
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        conn.execute("DELETE FROM crm_expert_session WHERE session_id=?", (token,))
        conn.commit()
    return {"ok": True}


def resolve_expert_session(headers: Any) -> dict[str, Any] | None:
    """Вернуть эксперта по X-Expert-Session или None."""
    try:
        token = (headers.get(SESSION_HEADER) or headers.get("X-Expert-Session") or "").strip()
    except Exception:  # noqa: BLE001
        token = ""
    if not token:
        return None
    ensure_expert_schema()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        row = conn.execute(
            """SELECT s.session_id, s.expires_at, u.expert_id, u.login, u.display_name, u.active
               FROM crm_expert_session s
               JOIN crm_expert_user u ON u.expert_id = s.expert_id
               WHERE s.session_id=?""",
            (token,),
        ).fetchone()
        if not row or not int(row["active"] or 0):
            return None
        if str(row["expires_at"]) < now:
            conn.execute("DELETE FROM crm_expert_session WHERE session_id=?", (token,))
            conn.commit()
            return None
        # sliding expiry
        expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
        conn.execute(
            """UPDATE crm_expert_session
               SET last_seen_at=?, expires_at=?
               WHERE session_id=?""",
            (
                now,
                expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                token,
            ),
        )
        conn.commit()
    return {
        "session_id": str(row["session_id"]),
        "expert_id": str(row["expert_id"]),
        "login": str(row["login"]),
        "display_name": str(row["display_name"] or row["login"]),
        "role": EXPERT_ROLE,
        "actor": f"expert:{row['login']}",
    }


def expert_path_allowed(path: str) -> bool:
    normalized = str(path or "")
    return any(normalized.startswith(prefix) for prefix in EXPERT_ALLOWED_PREFIXES)


def expert_status() -> dict[str, Any]:
    ensure_bootstrap_expert()
    ensure_expert_schema()
    with closing(_connect()) as conn:
        ensure_expert_schema(conn)
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM crm_expert_user WHERE active=1"
        ).fetchone()["n"]
    return {
        "ok": True,
        "role": EXPERT_ROLE,
        "active_users": int(count or 0),
        "reports_min_date": reports_min_date(),
        "bootstrap_ready": bool((os.environ.get("MO_EXPERT_BOOTSTRAP_PASSWORD") or "").strip())
        or int(count or 0) > 0,
    }
