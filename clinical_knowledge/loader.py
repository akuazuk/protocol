"""Загрузка реестра протоколов, нозологий и правил для consult-review."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GASTRO_MVP = ROOT / "data" / "gastro_mvp"
CATALOG_DIR = ROOT / "data" / "catalog"


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
        ROOT / "output" / "registry" / "protocol_cards.jsonl",
        GASTRO_MVP / "protocol_registry.jsonl",
    ]
    for p in paths:
        rows = _read_jsonl(p)
        if rows:
            return rows
    return []


def _merge_rules_from_dir(rules_dir: Path, out: dict[str, list[dict[str, Any]]]) -> None:
    if not rules_dir.is_dir():
        return
    for p in sorted(rules_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(data.get("condition_id") or "")
        if not cid:
            stem = p.stem.replace("_rules", "")
            for prefix in ("auto_", "path_", "enriched_"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix) :]
                    break
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
    out: dict[str, list[dict[str, Any]]] = {}
    _merge_rules_from_dir(GASTRO_MVP / "rules", out)
    _merge_rules_from_dir(CATALOG_DIR / "rules", out)
    try:
        from .rules_from_enrichment import load_enrichment_rules

        for cid, rules in load_enrichment_rules().items():
            bucket = out.setdefault(cid, [])
            seen = {r.get("rule_id") for r in bucket}
            for r in rules:
                rid = r.get("rule_id")
                if rid and rid in seen:
                    continue
                bucket.append(r)
                if rid:
                    seen.add(rid)
    except Exception:
        pass
    return out


def clear_clinical_knowledge_cache() -> None:
    load_protocol_cards_registry.cache_clear()
    load_conditions.cache_clear()
    load_rules_by_condition.cache_clear()


def clinical_knowledge_status() -> dict[str, Any]:
    from .coverage import coverage_status_payload

    cards = load_protocol_cards_registry()
    conditions = load_conditions()
    rules = load_rules_by_condition()
    rule_count = sum(len(v) for v in rules.values())
    enrichment_dir = GASTRO_MVP / "enrichment"
    enrichment_files = len(list(enrichment_dir.glob("*.json"))) if enrichment_dir.is_dir() else 0
    coverage = coverage_status_payload()
    return {
        "enabled": rule_count > 0,
        "protocol_cards": len(cards),
        "conditions": len(conditions),
        "rules": rule_count,
        "condition_ids_with_rules": len(rules),
        "mvp_scope": "all_catalog",
        "rules_coverage": coverage,
        "llm_enrichment_cached": enrichment_files,
    }
