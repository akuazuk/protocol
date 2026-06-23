"""SHA256 fingerprint for protocol PDF change detection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "protocol_catalog.jsonl"
FINGERPRINTS = ROOT / "data" / "protocol_summaries" / "source_fingerprints.jsonl"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def catalog_sha256(path: str) -> str | None:
    norm = path.replace("\\", "/")
    if not CATALOG.is_file():
        return None
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("path") or "").replace("\\", "/") == norm:
                return str(row.get("sha256") or "") or None
    return None


def load_fingerprints() -> dict[str, str]:
    out: dict[str, str] = {}
    if not FINGERPRINTS.is_file():
        return out
    with FINGERPRINTS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(row.get("protocol_id") or "")
            sha = str(row.get("sha256") or "")
            if pid and sha:
                out[pid] = sha
    return out


def save_fingerprint(protocol_id: str, sha256: str, path: str) -> None:
    FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if FINGERPRINTS.is_file():
        with FINGERPRINTS.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = str(row.get("protocol_id") or "")
                if pid:
                    rows[pid] = row
    rows[protocol_id] = {"protocol_id": protocol_id, "path": path, "sha256": sha256}
    with FINGERPRINTS.open("w", encoding="utf-8") as f:
        for pid in sorted(rows):
            f.write(json.dumps(rows[pid], ensure_ascii=False) + "\n")


def is_stale(protocol_id: str, current_sha256: str | None) -> bool:
    if not current_sha256:
        return False
    prev = load_fingerprints().get(protocol_id)
    return bool(prev and prev != current_sha256)
