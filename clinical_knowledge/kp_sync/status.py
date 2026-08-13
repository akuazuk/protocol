"""Свежий kp_sync_*.json для API и /api/corpus-stats (без ПДн)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sync_dirs() -> list[Path]:
    raw = [
        os.environ.get("KP_SYNC_DIR") or "",
        os.environ.get("PROTOCOL_CORPUS_ROOT") or "",
        "/var/data/protocol_corpus/_sync",
        str(ROOT / "data" / "kp_sync"),
    ]
    out: list[Path] = []
    for item in raw:
        item = str(item).strip()
        if not item:
            continue
        p = Path(item)
        if p.name != "_sync" and (p / "_sync").is_dir():
            p = p / "_sync"
        out.append(p)
    return out


def load_latest_kp_sync(sync_dir: Path | None = None) -> dict[str, Any] | None:
    dirs = [sync_dir] if sync_dir is not None else sync_dirs()
    files: list[Path] = []
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        files.extend(sorted(d.glob("kp_sync_*.json")))
    if not files:
        return None
    path = max(files, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    data["_source_file"] = path.name
    return data


def public_kp_sync_payload(raw: dict[str, Any] | None, *, days: int = 30) -> dict[str, Any]:
    if not raw:
        return {
            "ok": True,
            "status": "missing",
            "detail": "Сверки КП ещё не было",
            "added": [],
            "updated": [],
            "superseded": [],
            "changed_n": 0,
            "site_count": 0,
            "local_count": 0,
        }

    def _rows(items: list) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in items or []:
            if not isinstance(rec, dict):
                continue
            out.append(
                {
                    "filename": rec.get("filename") or "",
                    "slug": rec.get("slug") or rec.get("category") or "",
                    "relative_path": rec.get("relative_path") or "",
                    "action": rec.get("action") or "",
                    "alias_of": rec.get("alias_of") or "",
                }
            )
        return out[:200]

    added = _rows(raw.get("added") or [])
    updated = _rows(raw.get("updated") or [])
    superseded = _rows(raw.get("superseded") or [])
    changed_n = len(raw.get("changed_paths") or added + updated)
    return {
        "ok": True,
        "status": str(raw.get("status") or "success"),
        "crawled_utc": raw.get("crawled_utc") or raw.get("applied_utc") or "",
        "source_file": raw.get("_source_file") or "",
        "site_count": int(raw.get("site_count") or 0),
        "local_count": int(raw.get("local_count") or 0),
        "changed_n": changed_n,
        "added": added,
        "updated": updated,
        "superseded": superseded,
        "pending_summaries": int(raw.get("pending_summaries") or 0),
        "kp_corpus_generation": raw.get("kp_corpus_generation") or "",
        "rescored_n": int(raw.get("rescored_n") or 0),
        "days": days,
    }
