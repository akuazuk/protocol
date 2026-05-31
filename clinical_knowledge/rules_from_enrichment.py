"""Правила из кэша LLM-enrichment (gastro + catalog enrichment/*.json)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENRICH_DIRS = (
    ROOT / "data" / "gastro_mvp" / "enrichment",
    ROOT / "data" / "catalog" / "enrichment",
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (s or "").lower()).strip("_")[:40]


def enrichment_payload_to_rules(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Преобразовать один enrichment JSON в список правил."""
    cid = str(payload.get("condition_id") or "")
    enrich = payload.get("enrichment") or {}
    if not isinstance(enrich, dict):
        return []
    components = [
        str(x).strip()
        for x in (enrich.get("diagnosis_required_components") or [])
        if str(x).strip()
    ]
    if not components:
        return []
    src_path = payload.get("source_path")
    rules: list[dict[str, Any]] = [
        {
            "rule_id": f"llm_{_slug(cid)}_{payload.get('text_hash', 'x')[:8]}_diagnosis_formula",
            "rule_type": "diagnosis_formula",
            "required_components": components[:10],
            "severity": "warning",
            "description_ru": f"LLM-enrich: полнота формулировки диагноза ({cid}).",
            "source": {
                "source_path": src_path,
                "text_hash": payload.get("text_hash"),
                "llm_enriched": True,
            },
            "auto_extracted": True,
            "extraction_method": "llm_enrichment",
        }
    ]
    crit_summary = str(enrich.get("diagnostic_criteria_summary") or "").strip()
    if crit_summary:
        rules.append(
            {
                "rule_id": f"llm_{_slug(cid)}_{payload.get('text_hash', 'x')[:8]}_criteria_note",
                "rule_type": "diagnostic_criterion",
                "logic": "reference_only",
                "criteria": [{"summary": crit_summary[:500]}],
                "severity": "info",
                "description_ru": crit_summary[:300],
                "source": {"llm_enriched": True, "source_path": src_path},
                "auto_extracted": True,
                "extraction_method": "llm_enrichment",
            }
        )
    return rules


def load_enrichment_rules() -> dict[str, list[dict[str, Any]]]:
    """Все правила из кэша enrichment/."""
    out: dict[str, list[dict[str, Any]]] = {}
    for enrich_dir in ENRICH_DIRS:
        if not enrich_dir.is_dir():
            continue
        for p in sorted(enrich_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            cid = str(payload.get("condition_id") or "")
            if not cid:
                continue
            for rule in enrichment_payload_to_rules(payload):
                bucket = out.setdefault(cid, [])
                rid = rule.get("rule_id")
                if rid and any(r.get("rule_id") == rid for r in bucket):
                    continue
                bucket.append(rule)
    return out
