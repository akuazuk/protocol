"""Offline/online индекс профилей КП по path (обследование, лечение, наблюдение)."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from clinical_knowledge.catalog_build import resolve_chunks_path
from clinical_knowledge.protocol_icd_profile import build_protocol_icd_profile, merge_protocol_profiles

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "data" / "catalog" / "protocol_icd_profiles.jsonl"


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_OBLIGATION_RE = re.compile(r"обязательн|минимальн|must|необходим", re.I)


def _obligation_from_chunk(ch: dict[str, Any], line: str) -> str:
    text = (ch.get("text") or "") + " " + line
    if _OBLIGATION_RE.search(text):
        return "required"
    return "recommended"


def _profile_entry(path: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    icd_set: set[str] = set()
    for ch in chunks:
        for c in ch.get("icd10_codes") or []:
            icd_set.add(str(c).upper())
        for k in (ch.get("icd10_weights") or {}):
            icd_set.add(str(k).upper())
    icd_list = sorted(icd_set)[:24]
    prof = build_protocol_icd_profile(chunks, icd_list, path=path, query=" ".join(icd_list[:6]))

    def _tag_items(field: str, chunk_types: tuple[str, ...]) -> list[dict[str, Any]]:
        typed = [
            ch for ch in chunks
            if (ch.get("chunk_type") or ch.get("kind") or "").strip().lower() in chunk_types
        ]
        items: list[dict[str, Any]] = []
        for line in prof.get(field) or []:
            src = typed[0] if typed else {}
            items.append({
                "text": line,
                "obligation": _obligation_from_chunk(src, line),
            })
        return items[:20]

    return {
        "path": path,
        "icd_codes": icd_list,
        "diagnostics": _tag_items("diagnostics", ("diagnostics", "criteria_block", "table")),
        "medications": _tag_items("medications", ("pharmacotherapy", "drug_list")),
        "treatment": _tag_items("treatment", ("treatment", "pharmacotherapy")),
        "monitoring": _tag_items("monitoring", ("dispensary", "prevention")),
        "cites": prof.get("cites") or [],
    }


def build_protocol_icd_profile_index(
    *,
    chunks_path: Path | None = None,
    out_path: Path | None = None,
    max_paths: int | None = None,
) -> dict[str, Any]:
    """Один проход по rich_chunks: JSONL path → профиль КП."""
    from clinical_knowledge.catalog_build import build_chunks_index

    cp = chunks_path or resolve_chunks_path()
    out = out_path or DEFAULT_INDEX
    if not cp.is_file():
        return {"ok": False, "error": f"chunks not found: {cp}"}
    index = build_chunks_index(cp)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for path, chunks in sorted(index.items()):
            if max_paths and n >= max_paths:
                break
            if not chunks:
                continue
            entry = _profile_entry(path, chunks)
            if not (
                entry.get("diagnostics")
                or entry.get("medications")
                or entry.get("treatment")
            ):
                continue
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "paths_indexed": n, "index_path": str(out.relative_to(ROOT))}


@lru_cache(maxsize=1)
def _load_profile_index_by_path() -> dict[str, dict[str, Any]]:
    path = Path(os.environ.get("PROTOCOL_ICD_PROFILE_INDEX", str(DEFAULT_INDEX)))
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = (row.get("path") or "").replace("\\", "/")
            if sp:
                out[sp] = row
    return out


def _index_profile_to_merge_shape(entry: dict[str, Any]) -> dict[str, Any]:
    def _texts(items: list[Any]) -> list[str]:
        out: list[str] = []
        for it in items or []:
            if isinstance(it, dict):
                out.append(str(it.get("text") or ""))
            else:
                out.append(str(it))
        return [t for t in out if t]

    return {
        "path": entry.get("path"),
        "diagnostics": _texts(entry.get("diagnostics")),
        "medications": _texts(entry.get("medications")),
        "treatment": _texts(entry.get("treatment")),
        "monitoring": _texts(entry.get("monitoring")),
        "diagnostics_meta": entry.get("diagnostics") or [],
        "medications_meta": entry.get("medications") or [],
        "cites": entry.get("cites") or [],
    }


def merge_profiles_with_index(
    paths: list[str],
    icd_codes: list[str],
    get_chunks: Callable[[str], list[dict[str, Any]]],
    *,
    query: str = "",
) -> dict[str, Any]:
    """Профиль КП: offline index + fallback на live rich-чанки."""
    if not _env_bool("PROTOCOL_ICD_PROFILE_INDEX_ENABLED", True):
        return merge_protocol_profiles(paths, icd_codes, get_chunks, query=query)

    idx = _load_profile_index_by_path()
    if not idx:
        return merge_protocol_profiles(paths, icd_codes, get_chunks, query=query)

    merged = merge_protocol_profiles([], icd_codes, get_chunks, query=query)
    merged["paths"] = []
    seen_diag: set[str] = set()
    seen_med: set[str] = set()
    seen_treat: set[str] = set()
    seen_mon: set[str] = set()

    def _add(bucket: str, items: list[str], seen: set[str]) -> None:
        for it in items:
            key = re.sub(r"\s+", " ", (it or "").lower())[:80]
            if not key or key in seen:
                continue
            seen.add(key)
            merged.setdefault(bucket, []).append(it)

    from clinical_knowledge.consult_memory import consult_max_paths

    for path in paths[: consult_max_paths()]:
        entry = idx.get(path.replace("\\", "/"))
        if entry:
            prof = _index_profile_to_merge_shape(entry)
            merged["paths"].append(path)
            _add("diagnostics", prof.get("diagnostics") or [], seen_diag)
            _add("medications", prof.get("medications") or [], seen_med)
            _add("treatment", prof.get("treatment") or [], seen_treat)
            _add("monitoring", prof.get("monitoring") or [], seen_mon)
            for cite in prof.get("cites") or []:
                if len(merged.get("cites") or []) < 6:
                    merged.setdefault("cites", []).append(cite)
            merged.setdefault("diagnostics_meta", []).extend(entry.get("diagnostics") or [])
        else:
            from clinical_knowledge.consult_memory import consult_forbid_full_corpus

            if consult_forbid_full_corpus():
                if path not in merged["paths"]:
                    merged["paths"].append(path)
                continue
            live = merge_protocol_profiles([path], icd_codes, get_chunks, query=query)
            if path not in merged["paths"]:
                merged["paths"].append(path)
            for bucket, seen in (
                ("diagnostics", seen_diag),
                ("medications", seen_med),
                ("treatment", seen_treat),
                ("monitoring", seen_mon),
            ):
                _add(bucket, live.get(bucket) or [], seen)

    merged["index_used"] = bool(idx)
    return merged
