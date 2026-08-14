"""Пути данных rceth (GCE: /var/data/rceth)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/var/data/rceth")


def data_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = (os.environ.get("RCETH_DATA_ROOT") or "").strip()
    if env:
        return Path(env)
    if DEFAULT_DATA.is_dir() or os.environ.get("RCETH_USE_VAR_DATA", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return DEFAULT_DATA
    return ROOT / "data" / "rceth"


def sync_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "_sync"


def pdf_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "pdfs" / "instr"


def html_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "html" / "details"


def labels_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "labels"


def manifest_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "manifest.jsonl"


def status_path(root: Path | None = None) -> Path:
    return sync_dir(root) / "status.json"
