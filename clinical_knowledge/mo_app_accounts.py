"""Учётные записи МО Аналитики: логин/пароль, доступ к отчётам и период.

См. docs/plans/2026-08-09-auth-accounts-unify-v1.md (P1/P2).
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from .mo_backend import _connect, _utc

SESSION_TTL_HOURS = 12
SESSION_HEADER = "x-methodist-session"
REPORTS_MIN_DATE_DEFAULT = "2026-08-01"
ROLES = frozenset({"viewer", "methodist", "admin"})
MO_ACCESS = frozenset({"reports", "full"})

# API prefixes for mo_access=reports (как у бывшего expert).
REPORTS_ALLOWED_PREFIXES = (
    "/api/methodist/account/",
    "/api/methodist/mo/capabilities",
    "/api/methodist/mo/daily-report",
    "/api/methodist/mo/reports",
    "/api/methodist/mo/cases",
    "/api/methodist/mo/calibration",
    "/api/methodist/mo/review-packs",
    "/api/methodist/mo/rubric-summary",
    "/api/methodist/mo/freshness",
    "/api/methodist/mo/health",
    "/api/methodist/mo/meta",
)

APP_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_app_user (
  user_id TEXT PRIMARY KEY,
  login TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  display_name TEXT,
  role TEXT NOT NULL DEFAULT 'methodist',
  mo_access TEXT NOT NULL DEFAULT 'reports',
  reports_min_date TEXT NOT NULL DEFAULT '2026-08-01',
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_app_session (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_crm_app_session_user
  ON crm_app_session(user_id, expires_at);
"""

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def ensure_app_accounts_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    db = conn or _connect()
    try:
        db.executescript(APP_SCHEMA_SQL)
        if own:
            db.commit()
    finally:
        if own:
            db.close()


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


def _normalize_date(value: str | None, *, default: str = REPORTS_MIN_DATE_DEFAULT) -> str:
    raw = str(value or "").strip()[:10]
    if _DATE_RE.match(raw):
        return raw
    return default


def _row_public(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    get = row.__getitem__ if hasattr(row, "__getitem__") else row.get  # type: ignore[assignment]
    return {
        "user_id": str(get("user_id")),
        "login": str(get("login")),
        "display_name": str(get("display_name") or get("login")),
        "role": str(get("role") or "methodist"),
        "mo_access": str(get("mo_access") or "reports"),
        "reports_min_date": str(get("reports_min_date") or REPORTS_MIN_DATE_DEFAULT),
        "active": bool(int(get("active") or 0)),
        "created_at": str(get("created_at") or ""),
        "updated_at": str(get("updated_at") or ""),
    }


def list_users(*, include_inactive: bool = True) -> dict[str, Any]:
    ensure_app_accounts_schema()
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        if include_inactive:
            rows = conn.execute(
                "SELECT * FROM crm_app_user ORDER BY created_at DESC, login ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM crm_app_user WHERE active=1 ORDER BY created_at DESC, login ASC"
            ).fetchall()
    return {"ok": True, "users": [_row_public(r) for r in rows], "n": len(rows)}


def upsert_user(
    *,
    login: str,
    password: str | None = None,
    display_name: str = "",
    role: str = "methodist",
    mo_access: str = "reports",
    reports_min_date: str | None = None,
    active: bool = True,
    user_id: str | None = None,
) -> dict[str, Any]:
    login_norm = str(login or "").strip().lower()
    role_norm = str(role or "methodist").strip().lower()
    if role_norm not in ROLES:
        raise ValueError("invalid_role")
    access_norm = str(mo_access or "reports").strip().lower()
    if access_norm not in MO_ACCESS:
        raise ValueError("invalid_mo_access")
    min_date = _normalize_date(reports_min_date)
    now = _utc()
    ensure_app_accounts_schema()
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        existing = None
        if user_id:
            existing = conn.execute(
                "SELECT * FROM crm_app_user WHERE user_id=?",
                (str(user_id),),
            ).fetchone()
            if existing is None:
                raise ValueError("user_not_found")
        if existing is None and login_norm:
            existing = conn.execute(
                "SELECT * FROM crm_app_user WHERE login=?",
                (login_norm,),
            ).fetchone()
        if existing:
            uid = str(existing["user_id"])
            if not login_norm:
                login_norm = str(existing["login"])
            if len(login_norm) < 2:
                raise ValueError("login_required")
            pwd = str(existing["password_hash"])
            if password is not None and str(password).strip():
                if len(str(password)) < 8:
                    raise ValueError("password_too_short")
                pwd = _hash_password(str(password))
            name = (display_name or existing["display_name"] or login_norm)[:120]
            conn.execute(
                """UPDATE crm_app_user
                   SET login=?, password_hash=?, display_name=?, role=?, mo_access=?,
                       reports_min_date=?, active=?, updated_at=?
                   WHERE user_id=?""",
                (
                    login_norm,
                    pwd,
                    name,
                    role_norm,
                    access_norm,
                    min_date,
                    1 if active else 0,
                    now,
                    uid,
                ),
            )
        else:
            if not login_norm or len(login_norm) < 2:
                raise ValueError("login_required")
            if not password or len(str(password)) < 8:
                raise ValueError("password_too_short")
            uid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO crm_app_user(
                     user_id, login, password_hash, display_name, role, mo_access,
                     reports_min_date, active, created_at, updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    uid,
                    login_norm,
                    _hash_password(str(password)),
                    (display_name or login_norm)[:120],
                    role_norm,
                    access_norm,
                    min_date,
                    1 if active else 0,
                    now,
                    now,
                ),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM crm_app_user WHERE user_id=?", (uid,)).fetchone()
    out = _row_public(row)
    out["ok"] = True
    return out


def set_user_active(user_id: str, *, active: bool) -> dict[str, Any]:
    ensure_app_accounts_schema()
    now = _utc()
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        cur = conn.execute(
            "UPDATE crm_app_user SET active=?, updated_at=? WHERE user_id=?",
            (1 if active else 0, now, str(user_id)),
        )
        if cur.rowcount <= 0:
            raise ValueError("user_not_found")
        if not active:
            conn.execute("DELETE FROM crm_app_session WHERE user_id=?", (str(user_id),))
        conn.commit()
        row = conn.execute(
            "SELECT * FROM crm_app_user WHERE user_id=?", (str(user_id),)
        ).fetchone()
    out = _row_public(row)
    out["ok"] = True
    return out


def login_user(*, login: str, password: str) -> dict[str, Any]:
    login_norm = str(login or "").strip().lower()
    if not login_norm or not password:
        raise PermissionError("invalid_credentials")
    ensure_app_accounts_schema()
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        row = conn.execute(
            "SELECT * FROM crm_app_user WHERE login=? AND active=1",
            (login_norm,),
        ).fetchone()
        if not row or not _verify_password(password, str(row["password_hash"])):
            raise PermissionError("invalid_credentials")
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=SESSION_TTL_HOURS)
        conn.execute(
            """INSERT INTO crm_app_session(
                 session_id, user_id, created_at, expires_at, last_seen_at
               ) VALUES (?,?,?,?,?)""",
            (
                session_id,
                row["user_id"],
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            ),
        )
        conn.commit()
    public = _row_public(row)
    return {
        "ok": True,
        "session_token": session_id,
        "expires_in_hours": SESSION_TTL_HOURS,
        **public,
    }


def logout_user(session_token: str) -> dict[str, Any]:
    token = str(session_token or "").strip()
    if not token:
        return {"ok": True}
    ensure_app_accounts_schema()
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        conn.execute("DELETE FROM crm_app_session WHERE session_id=?", (token,))
        conn.commit()
    return {"ok": True}


def resolve_app_session(headers: Any) -> dict[str, Any] | None:
    try:
        token = (headers.get(SESSION_HEADER) or headers.get("X-Methodist-Session") or "").strip()
    except Exception:  # noqa: BLE001
        token = ""
    if not token:
        return None
    ensure_app_accounts_schema()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with closing(_connect()) as conn:
        ensure_app_accounts_schema(conn)
        row = conn.execute(
            """SELECT s.session_id, s.expires_at, u.*
               FROM crm_app_session s
               JOIN crm_app_user u ON u.user_id = s.user_id
               WHERE s.session_id=?""",
            (token,),
        ).fetchone()
        if not row or not int(row["active"] or 0):
            return None
        if str(row["expires_at"]) < now:
            conn.execute("DELETE FROM crm_app_session WHERE session_id=?", (token,))
            conn.commit()
            return None
        expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
        conn.execute(
            """UPDATE crm_app_session
               SET last_seen_at=?, expires_at=?
               WHERE session_id=?""",
            (
                now,
                expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                token,
            ),
        )
        conn.commit()
    public = _row_public(row)
    public["session_id"] = str(row["session_id"])
    public["actor"] = f"user:{public['login']}"
    return public


def path_allowed_for_access(path: str, mo_access: str) -> bool:
    access = str(mo_access or "reports").strip().lower()
    if access == "full":
        return True
    normalized = str(path or "")
    return any(normalized.startswith(prefix) for prefix in REPORTS_ALLOWED_PREFIXES)


def user_reports_min_date(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    return _normalize_date(user.get("reports_min_date"))


def capabilities_for_user(user: dict[str, Any]) -> dict[str, Any]:
    """Capabilities с учётом mo_access и персонального периода отчётов."""
    from .mo_backend import build_mo_capabilities

    role = str(user.get("role") or "methodist")
    access = str(user.get("mo_access") or "reports")
    # reports-доступ рисуем как expert-страницы, но роль оставляем реальной.
    caps_role = "expert" if access == "reports" else role
    caps = build_mo_capabilities(caps_role)
    caps["role"] = role
    caps["mo_access"] = access
    caps["reports_min_date"] = user_reports_min_date(user)
    if access == "reports":
        # Явно зафиксировать набор страниц отчётов.
        pages = caps.get("pages") or {}
        for key in list(pages.keys()):
            pages[key] = key in {"yesterday", "reports"}
        caps["pages"] = pages
    return caps
