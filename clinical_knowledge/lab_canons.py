"""Каноны лабораторных панелей из data/lab_canons/lab_test_canons.json."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_CANON_PATH = _ROOT / "data" / "lab_canons" / "lab_test_canons.json"
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)


def _norm(text: Any) -> str:
    raw = str(text or "").lower().replace("ё", "е")
    return _WS.sub(" ", raw).strip()


@lru_cache(maxsize=1)
def load_lab_canons() -> dict[str, Any]:
    if not _CANON_PATH.is_file():
        return {"version": "", "panels": []}
    return json.loads(_CANON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def lab_panels() -> tuple[dict[str, Any], ...]:
    raw = load_lab_canons().get("panels") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        patterns = [
            re.compile(p, re.I)
            for p in (item.get("text_patterns") or [])
            if p
        ]
        out.append(
            {
                "id": str(item["id"]),
                "label": str(item.get("label") or item["id"]),
                "type_needles": tuple(
                    _norm(n) for n in (item.get("type_needles") or []) if n
                ),
                "text_patterns": tuple(patterns),
                "indicators": tuple(
                    _norm(n) for n in (item.get("indicators") or []) if n
                ),
            }
        )
    return tuple(out)


def panel_by_id(panel_id: str) -> dict[str, Any] | None:
    for panel in lab_panels():
        if panel["id"] == panel_id:
            return panel
    return None


def text_hits_panel(text: str, panel: dict[str, Any]) -> bool:
    blob = _norm(text)
    if not blob:
        return False
    for rx in panel.get("text_patterns") or ():
        if rx.search(blob):
            return True
    for needle in panel.get("type_needles") or ():
        if needle and needle in blob:
            return True
    for needle in panel.get("indicators") or ():
        if needle and needle in blob:
            return True
    return False


def type_hits_panel(type_name: str, panel: dict[str, Any]) -> bool:
    n = _norm(type_name)
    if not n:
        return False
    tokens = set(_TOKEN.findall(n))
    for needle in panel.get("type_needles") or ():
        if not needle:
            continue
        if " " not in needle and len(needle) <= 4:
            if needle in tokens:
                return True
            continue
        if needle in n:
            return True
    for needle in panel.get("indicators") or ():
        if needle and needle in n:
            return True
    return False


def panels_mentioned_in_text(text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for panel in lab_panels():
        if text_hits_panel(text, panel):
            out[panel["id"]] = {"id": panel["id"], "label": panel["label"]}
    return out


def clear_lab_canon_cache() -> None:
    load_lab_canons.cache_clear()
    lab_panels.cache_clear()
