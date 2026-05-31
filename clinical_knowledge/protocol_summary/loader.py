"""Загрузка Protocol Summary Cards с диска."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..rule_model import ProtocolRule
from . import config as _cfg_mod
from .config import protocol_summary_config
from .schema import ConditionSummary, ProtocolSummary
from .summary_to_rules import summary_to_protocol_rules
from .validator import summary_is_usable

ROOT = Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    p = Path(_cfg_mod.protocol_summary_config.data_root)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _summary_search_dirs() -> list[Path]:
    root = _data_root()
    return [
        root / "reviewed",
        root / "yaml",
        root / "json",
        root / "drafts",
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("PyYAML required for protocol summaries") from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_summary_file(path: Path) -> ProtocolSummary | None:
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = _load_yaml(path)
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            return None
        return ProtocolSummary.model_validate(data)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_protocol_summaries(*, usable_only: bool = False) -> tuple[ProtocolSummary, ...]:
    seen: dict[str, ProtocolSummary] = {}
    for d in _summary_search_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*")):
            if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            summary = _parse_summary_file(path)
            if summary is None:
                continue
            pid = summary.protocol_id
            if pid not in seen:
                seen[pid] = summary
    out = list(seen.values())
    if usable_only:
        out = [s for s in out if summary_is_usable(s)]
    return tuple(out)


def clear_protocol_summary_cache() -> None:
    load_protocol_summaries.cache_clear()
    load_summary_by_protocol_id.cache_clear()
    load_summary_rules.cache_clear()
    _summaries_by_condition_id.cache_clear()


@lru_cache(maxsize=256)
def load_summary_by_protocol_id(protocol_id: str) -> ProtocolSummary | None:
    for d in _summary_search_dirs():
        for ext in (".yaml", ".yml", ".json"):
            path = d / f"{protocol_id}{ext}"
            if path.is_file():
                return _parse_summary_file(path)
    return None


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


@lru_cache(maxsize=2)
def _summaries_by_condition_id(*, usable_only: bool = False) -> dict[str, tuple[ProtocolSummary, ConditionSummary]]:
    out: dict[str, tuple[ProtocolSummary, ConditionSummary]] = {}
    for summary in load_protocol_summaries(usable_only=usable_only):
        for cond in summary.conditions:
            out[cond.condition_id] = (summary, cond)
    return out


def find_summary_for_condition(
    condition: ConditionSummary,
    *,
    usable_only: bool = False,
) -> ProtocolSummary | None:
    entry = _summaries_by_condition_id(usable_only=usable_only).get(condition.condition_id)
    if entry is None:
        return None
    return entry[0]


def find_conditions_by_icd(icd10_code: str) -> list[ConditionSummary]:
    code = _norm_icd(icd10_code)
    if not code:
        return []
    found: list[ConditionSummary] = []
    for _summary, cond in _summaries_by_condition_id(usable_only=False).values():
        codes = {_norm_icd(c) for c in cond.icd10_codes}
        if code in codes or any(code.startswith(c) or c.startswith(code) for c in codes if c):
            found.append(cond)
    return found


def find_conditions_by_text(query: str, *, limit: int = 20) -> list[ConditionSummary]:
    q = re.sub(r"\s+", " ", (query or "").lower().strip())
    if len(q) < 2:
        return []
    found: list[ConditionSummary] = []
    for _summary, cond in _summaries_by_condition_id(usable_only=False).values():
        blob = " ".join(
            [cond.name.lower()]
            + [s.lower() for s in cond.synonyms]
            + [c.lower() for c in cond.icd10_codes]
        )
        if q in blob or any(q in s for s in cond.synonyms):
            found.append(cond)
            if len(found) >= limit:
                return found
    return found


@lru_cache(maxsize=1)
def load_summary_rules(*, usable_only: bool = True) -> tuple[ProtocolRule, ...]:
    rules: list[ProtocolRule] = []
    for summary in load_protocol_summaries(usable_only=usable_only):
        rules.extend(summary_to_protocol_rules(summary))
    return tuple(rules)


def export_summary_json(summary: ProtocolSummary, out_dir: Path | None = None) -> Path:
    root = _data_root()
    out_dir = out_dir or (root / "json")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.protocol_id}.json"
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
