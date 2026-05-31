"""Загрузка реестра протоколов, нозологий и правил для consult-review."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GASTRO_MVP = ROOT / "data" / "gastro_mvp"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@lru_cache(maxsize=1)
def load_protocol_cards_registry() -> list[dict[str, Any]]:
    paths = [
        GASTRO_MVP / "protocol_registry.jsonl",
        ROOT / "output" / "registry" / "protocol_cards.jsonl",
    ]
    for p in paths:
        rows = _read_jsonl(p)
        if rows:
            return rows
    return []


@lru_cache(maxsize=1)
def load_conditions() -> dict[str, dict[str, Any]]:
    cond_dir = GASTRO_MVP / "conditions"
    out: dict[str, dict[str, Any]] = {}
    if not cond_dir.is_dir():
        return out
    for p in sorted(cond_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(data.get("condition_id") or p.stem)
        out[cid] = data
    return out


@lru_cache(maxsize=1)
def load_rules_by_condition() -> dict[str, list[dict[str, Any]]]:
    rules_dir = GASTRO_MVP / "rules"
    out: dict[str, list[dict[str, Any]]] = {}
    if not rules_dir.is_dir():
        return out
    for p in sorted(rules_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(data.get("condition_id") or "")
        if not cid:
            stem = p.stem.replace("_rules", "").replace("auto_", "")
            cid = stem
        rules = list(data.get("rules") or [])
        if not rules:
            continue
        bucket = out.setdefault(cid, [])
        seen = {r.get("rule_id") for r in bucket}
        for r in rules:
            rid = r.get("rule_id")
            if rid and rid in seen:
                continue
            bucket.append(r)
            if rid:
                seen.add(rid)
    return out


def clinical_knowledge_status() -> dict[str, Any]:
    cards = load_protocol_cards_registry()
    conditions = load_conditions()
    rules = load_rules_by_condition()
    rule_count = sum(len(v) for v in rules.values())
    return {
        "enabled": bool(conditions and rules),
        "protocol_cards": len(cards),
        "conditions": len(conditions),
        "rules": rule_count,
        "mvp_scope": "gastroenterologiya",
    }
