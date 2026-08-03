"""Дополнение правил каталога из rich-чанков (table, drug_list) для runtime и build."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .catalog_build import build_chunks_index, resolve_chunks_path
from .table_rule_extractor import rule_from_table_chunk

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _rule_to_legacy(rule: Any, *, condition_id: str = "rich_table") -> dict[str, Any]:
    src = rule.source
    rt = rule.rule_type
    if rt.endswith("_rule"):
        rt = rt[: -len("_rule")]
    return {
        "rule_id": rule.rule_id,
        "rule_type": rt,
        "severity": rule.severity,
        "condition_id": condition_id,
        "expected_items": list(rule.expected_items or []),
        "evidence_targets": list(rule.evidence_targets or []),
        "extraction_method": "rich_table_chunk",
        "rule_source": "rich_chunks",
        "source": {
            "source_path": src.local_path if src else None,
            "protocol_id": src.protocol_id if src else None,
            "page": src.page_start if src else None,
            "section_title": src.section_title if src else None,
            "quote": src.quote if src else None,
        },
    }


@lru_cache(maxsize=1)
def _chunks_index_cached() -> dict[str, list[dict[str, Any]]]:
    cp = resolve_chunks_path()
    if not cp.is_file():
        return {}
    return build_chunks_index(cp)


def rich_table_rules_for_paths(
    source_paths: list[str] | None,
    *,
    limit_per_path: int = 12,
) -> list[dict[str, Any]]:
    """Legacy-правила из table/drug_list rich-чанков для указанных PDF."""
    if not _env_bool("CONSULT_RICH_RULES", True):
        return []
    paths = [p.replace("\\", "/").strip() for p in (source_paths or []) if p]
    if not paths:
        return []
    index = _chunks_index_cached()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sp in paths:
        chunks = index.get(sp) or []
        n = 0
        for ch in chunks:
            if n >= limit_per_path:
                break
            ctype = (ch.get("chunk_type") or ch.get("kind") or "").strip().lower()
            if ctype not in ("table", "table_block", "drug_list", "criteria_block"):
                continue
            rule = rule_from_table_chunk(ch)
            if not rule or rule.rule_id in seen:
                continue
            seen.add(rule.rule_id)
            out.append(_rule_to_legacy(rule))
            n += 1
    return out


def merge_table_rules_into_catalog_extracted(
    extracted: dict[str, list[dict[str, Any]]],
    chunks_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Offline: добавить table-правила в результат extract_rules_all_catalog_pdfs."""
    cp = chunks_path or resolve_chunks_path()
    if not cp.is_file():
        return extracted
    merged = {k: list(v) for k, v in extracted.items()}
    seen: set[str] = set()
    for _sp, chunks in build_chunks_index(cp).items():
        for ch in chunks:
            ctype = (ch.get("chunk_type") or "").strip().lower()
            if ctype not in ("table", "table_block", "drug_list"):
                continue
            rule = rule_from_table_chunk(ch)
            if not rule:
                continue
            leg = _rule_to_legacy(rule, condition_id="rich_table")
            rid = leg.get("rule_id") or ""
            if rid in seen:
                continue
            seen.add(rid)
            merged.setdefault("rich_table", []).append(leg)
    return merged
