"""Загрузка snippet-паков по специальности (YAML)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPECIALTY_DIR = ROOT / "data" / "patient_specialty"
PENDING_DIR = SPECIALTY_DIR / "_pending"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def list_specialty_packs() -> list[str]:
    if not SPECIALTY_DIR.is_dir():
        return []
    return sorted(p.stem for p in SPECIALTY_DIR.glob("*.yaml"))


def load_specialty_pack(specialty: str | None) -> dict[str, Any]:
    spec = (specialty or "default").strip().lower()
    pack = _load_yaml(SPECIALTY_DIR / f"{spec}.yaml")
    if not pack and spec != "default":
        pack = _load_yaml(SPECIALTY_DIR / "default.yaml")
    return pack


def list_pending_snippet_updates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not PENDING_DIR.is_dir():
        return out
    for path in sorted(PENDING_DIR.glob("*.yaml"), reverse=True):
        data = _load_yaml(path)
        out.append(
            {
                "file": path.name,
                "path": str(path.relative_to(ROOT)),
                "specialty": data.get("specialty"),
                "summary_ru": data.get("summary_ru") or "",
                "created_at": data.get("created_at") or "",
                "changes": data.get("proposed_changes") or [],
            }
        )
    return out
