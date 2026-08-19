"""Индекс инструкций Rceth по INN (+ форма) для shadow label-check."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from clinical_knowledge.rceth_sync.identity import canon_inn, form_keywords
from clinical_knowledge.rceth_sync.paths import data_root, labels_dir


def _usable_label(rec: dict[str, Any]) -> bool:
    if not isinstance(rec, dict):
        return False
    if str(rec.get("status") or "active") != "active":
        return False
    parse = rec.get("parse") if isinstance(rec.get("parse"), dict) else {}
    if parse.get("extract_error"):
        return False
    sections = rec.get("sections") if isinstance(rec.get("sections"), dict) else {}
    return bool(
        sections.get("indications_4_1")
        or sections.get("contraindications_4_3")
        or sections.get("posology_4_2")
    )


def _label_inn(rec: dict[str, Any]) -> str | None:
    return canon_inn(str(rec.get("inn") or ""))


def _label_forms(rec: dict[str, Any]) -> list[str]:
    forms: list[str] = []
    raw = rec.get("forms")
    if isinstance(raw, list):
        forms.extend(str(x) for x in raw if x)
    forms.extend(form_keywords(str(rec.get("form_text") or "")))
    out: list[str] = []
    for item in forms:
        key = str(item).strip().lower()
        if key and key not in out:
            out.append(key)
    return out


def _iter_label_files(root: Path) -> list[Path]:
    labeled = labels_dir(root)
    files = sorted(labeled.glob("*.json")) if labeled.is_dir() else []
    if files:
        return files
    if root.is_dir():
        return sorted(p for p in root.glob("*.json") if p.name != "manifest.json")
    return []


def build_label_ctx(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_inn: dict[str, list[dict[str, Any]]] = {}
    for rec in rows:
        if not _usable_label(rec):
            continue
        inn = _label_inn(rec)
        if not inn:
            continue
        item = dict(rec)
        item["_forms"] = _label_forms(rec)
        by_inn.setdefault(inn, []).append(item)
    return {"by_inn": by_inn, "inn_count": len(by_inn)}


@lru_cache(maxsize=8)
def load_rceth_label_ctx(root: str | None = None) -> dict[str, Any]:
    data = data_root(root) if root else data_root()
    rows: list[dict[str, Any]] = []
    for path in _iter_label_files(Path(data)):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return build_label_ctx(rows)


def lookup_label(
    ctx: dict[str, Any] | None,
    inn: str,
    form: str | None = None,
) -> dict[str, Any] | None:
    if not ctx:
        return None
    key = canon_inn(inn) or str(inn or "").strip().lower()
    cands = list((ctx.get("by_inn") or {}).get(key) or [])
    if not cands:
        return None
    want = str(form or "").strip().lower()
    if want:
        matched = [r for r in cands if want in (r.get("_forms") or [])]
        if matched:
            return matched[0]
    return cands[0]


def clear_label_ctx_cache() -> None:
    load_rceth_label_ctx.cache_clear()
