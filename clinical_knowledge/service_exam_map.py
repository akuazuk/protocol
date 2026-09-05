"""Mapping услуг МИС → канон обследования КП (wave 3 stub)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "data" / "catalog" / "mis_service_exam_map.json"
_WS = re.compile(r"\s+")


def _norm(text: Any) -> str:
    return _WS.sub(" ", str(text or "").lower().replace("ё", "е")).strip()


@lru_cache(maxsize=1)
def load_service_exam_map() -> list[dict[str, Any]]:
    if not _PATH.is_file():
        return []
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    return list(data.get("mappings") or [])


def map_service_to_panel(service_name: str) -> dict[str, Any] | None:
    blob = _norm(service_name)
    if not blob:
        return None
    for row in load_service_exam_map():
        needles = [_norm(n) for n in (row.get("service_needles") or [])]
        if any(n and n in blob for n in needles):
            return {
                "panel_id": row.get("panel_id"),
                "protocol_exam": row.get("protocol_exam"),
                "service": service_name,
            }
    return None


def map_services(service_names: list[str] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in service_names or []:
        hit = map_service_to_panel(str(name or ""))
        if not hit:
            continue
        key = str(hit.get("panel_id") or hit.get("protocol_exam") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out
