"""Live status.json + итог прогона (зеркало kp_sync status pattern)."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_knowledge.rceth_sync.paths import status_path, sync_dir


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def write_status(
    *,
    phase: str,
    status: str = "running",
    done: int = 0,
    total: int = 0,
    message: str = "",
    current_reg_id: str = "",
    errors: int = 0,
    retries_503: int = 0,
    root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Обновить live status для UI poll."""
    path = status_path(root)
    prev: dict[str, Any] = {}
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
    started = prev.get("started_at") if prev.get("status") == "running" else None
    payload: dict[str, Any] = {
        "ok": True,
        "status": status,
        "phase": phase,
        "progress": {"done": int(done), "total": int(total)},
        "message": message,
        "current_reg_id": current_reg_id,
        "errors": int(errors),
        "retries_503": int(retries_503),
        "updated_at": _now(),
        "started_at": started or _now(),
        "pid": os.getpid(),
    }
    if status in {"done", "error", "idle"}:
        payload["finished_at"] = _now()
    if extra:
        payload.update(extra)
    _atomic_write(path, payload)
    return payload


def read_status(root: Path | None = None) -> dict[str, Any] | None:
    path = status_path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_sync_summary(
    summary: dict[str, Any],
    *,
    root: Path | None = None,
    day: str | None = None,
) -> Path:
    """Итог прогона rceth_sync_YYYY-MM-DD.json."""
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = sync_dir(root) / f"rceth_sync_{day}.json"
    payload = dict(summary)
    payload.setdefault("ok", True)
    payload["sync_day"] = day
    payload["written_at"] = _now()
    _atomic_write(out, payload)
    return out


def load_all_rceth_syncs(root: Path | None = None) -> list[dict[str, Any]]:
    """Все rceth_sync_YYYY-MM-DD.json по возрастанию дня."""
    folder = sync_dir(root)
    if not folder.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("rceth_sync_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        day = data.get("sync_day") or path.stem.replace("rceth_sync_", "", 1)
        data = dict(data)
        data["_sync_day"] = day
        data.setdefault("sync_day", day)
        rows.append(data)
    return rows


def load_latest_rceth_sync(root: Path | None = None) -> dict[str, Any] | None:
    rows = load_all_rceth_syncs(root)
    return rows[-1] if rows else None


def _parse_utc(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pid_alive(pid: Any) -> bool | None:
    """True/False if checkable; None if unknown."""
    try:
        n = int(pid)
    except (TypeError, ValueError):
        return None
    if n <= 1:
        return None
    try:
        os.kill(n, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def resolve_live_status(
    live: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    stale_sec: int | None = None,
) -> tuple[dict[str, Any] | None, bool, str]:
    """Нормализовать live: dead PID / старый heartbeat → interrupted, running=false.

    Returns: (live_view, running, top_status).
    """
    if not live:
        return None, False, "unavailable"
    view = dict(live)
    raw_status = str(view.get("status") or "")
    claimed = raw_status in {"running", "queued"}
    if not claimed:
        return view, False, raw_status or "idle"

    if stale_sec is None:
        try:
            stale_sec = int(os.environ.get("RCETH_STATUS_STALE_SEC", "180") or "180")
        except ValueError:
            stale_sec = 180
    stale_sec = max(30, stale_sec)
    now = now or datetime.now(timezone.utc)
    updated = _parse_utc(str(view.get("updated_at") or ""))
    age_sec = int((now - updated).total_seconds()) if updated else None
    pid_ok = _pid_alive(view.get("pid"))

    reason = ""
    if pid_ok is False:
        reason = "process_gone"
    elif age_sec is not None and age_sec > stale_sec and pid_ok is not True:
        reason = "heartbeat_stale"
    elif age_sec is not None and age_sec > max(stale_sec * 10, 1800) and pid_ok is True:
        # OCR на одном файле слишком долго - считаем зависшим даже при живом PID.
        reason = "heartbeat_stale_long"

    if reason:
        prog = view.get("progress") if isinstance(view.get("progress"), dict) else {}
        msg = (view.get("message") or "").strip()
        view["status"] = "interrupted"
        view["stale"] = True
        view["stale_reason"] = reason
        view["age_sec"] = age_sec
        view["message"] = (
            f"прерван ({reason})"
            + (f": {msg}" if msg else "")
            + (f" · last {prog.get('done')}/{prog.get('total')}" if prog else "")
        )
        view.setdefault("finished_at", view.get("updated_at") or _now())
        return view, False, "interrupted"

    return view, True, "running"


def public_rceth_sync_payload(
    latest: dict[str, Any] | None = None,
    live: dict[str, Any] | None = None,
    *,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Публичный снимок для /api/methodist/mo/rceth-sync (без ПДн)."""
    live = live if live is not None else read_status()
    if latest is None:
        latest = load_latest_rceth_sync()
    if history is None:
        history = load_all_rceth_syncs()
    live_view, running, top_status = resolve_live_status(live)
    if top_status == "unavailable" and latest:
        top_status = "idle"
    out: dict[str, Any] = {
        "ok": True,
        "status": top_status,
        "running": running,
        "live": None,
        "latest": None,
        "history": [],
    }
    if live_view:
        prog = live_view.get("progress") if isinstance(live_view.get("progress"), dict) else {}
        out["live"] = {
            "phase": live_view.get("phase"),
            "status": live_view.get("status"),
            "done": prog.get("done"),
            "total": prog.get("total"),
            "message": live_view.get("message") or "",
            "current_reg_id": live_view.get("current_reg_id") or "",
            "errors": live_view.get("errors") or 0,
            "retries_503": live_view.get("retries_503") or 0,
            "updated_at": live_view.get("updated_at"),
            "started_at": live_view.get("started_at"),
            "finished_at": live_view.get("finished_at"),
            "stale": bool(live_view.get("stale")),
            "stale_reason": live_view.get("stale_reason") or "",
            "age_sec": live_view.get("age_sec"),
        }
    if latest:
        out["latest"] = {
            "sync_day": latest.get("sync_day") or latest.get("_sync_day") or "",
            "manifest_count": latest.get("manifest_count"),
            "with_s_pdf": latest.get("with_s_pdf"),
            "downloaded": latest.get("downloaded"),
            "failed": latest.get("failed"),
            "no_pdf": latest.get("no_pdf"),
            "parse_ok": latest.get("parse_ok"),
            "written_at": latest.get("written_at"),
        }
        if top_status not in {"running", "queued", "interrupted", "error"}:
            out["status"] = "idle"
        out["sync_day"] = out["latest"]["sync_day"]
    hist_pub: list[dict[str, Any]] = []
    for row in history[-30:]:
        hist_pub.append(
            {
                "sync_day": row.get("sync_day") or row.get("_sync_day") or "",
                "manifest_count": row.get("manifest_count"),
                "with_s_pdf": row.get("with_s_pdf"),
                "downloaded": row.get("downloaded"),
                "failed": row.get("failed") or 0,
                "no_pdf": row.get("no_pdf"),
                "parse_ok": row.get("parse_ok"),
            }
        )
    out["history"] = hist_pub
    return out
