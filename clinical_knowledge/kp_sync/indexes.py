"""Merge catalog + ICD-профилей только для changed paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from clinical_knowledge.kp_sync.jsonl_merge import load_jsonl, merge_jsonl_by_path, write_jsonl
from clinical_knowledge.kp_sync.metadata import extract_protocol_metadata
from clinical_knowledge.protocol_icd_profile_index import DEFAULT_INDEX, _profile_entry

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "data" / "protocol_catalog.jsonl"


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/")


def merge_icd_profiles_for_paths(
    paths: list[str],
    *,
    chunks_by_path: dict[str, list[dict[str, Any]]],
    out_path: Path | None = None,
) -> dict[str, int]:
    dest = out_path or DEFAULT_INDEX
    incoming: list[dict[str, Any]] = []
    replace = {_norm(p) for p in paths if p}
    for path in sorted(replace):
        chunks = chunks_by_path.get(path) or []
        if not chunks:
            continue
        incoming.append(_profile_entry(path, chunks))
    old = load_jsonl(dest)
    merged = merge_jsonl_by_path(old, incoming)
    write_jsonl(dest, merged)
    return {"before": len(old), "incoming": len(incoming), "after": len(merged)}


def _icd_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ch in chunks:
        for code in ch.get("icd10_codes") or []:
            c = str(code).upper().strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out[:24]


def catalog_stub_for_path(
    path: str,
    *,
    chunks: list[dict[str, Any]] | None = None,
    text: str = "",
    filename: str = "",
) -> dict[str, Any]:
    fn = filename or Path(path).name
    meta = extract_protocol_metadata(text=text, filename=fn, source_path=path)
    icd = _icd_from_chunks(chunks or [])
    slug = ""
    parts = _norm(path).split("/")
    if len(parts) >= 2 and parts[0] == "minzdrav_protocols":
        slug = parts[1]
    return {
        "path": _norm(path),
        "filename": fn,
        "title": meta.get("title") or fn,
        "category": slug,
        "audience": meta.get("audience") or "any",
        "protocol_kind": meta.get("protocol_kind") or "clinical",
        "icd10_primary": icd[:12],
        "icd10_all": icd,
        "approval_date": meta.get("approval_date"),
        "approval_number": meta.get("approval_number"),
        "clinical_for_score": bool(meta.get("clinical_for_score")),
    }


def merge_catalog_for_paths(
    paths: list[str],
    *,
    chunks_by_path: dict[str, list[dict[str, Any]]],
    out_path: Path | None = None,
) -> dict[str, int]:
    dest = out_path or CATALOG_PATH
    replace = {_norm(p) for p in paths if p}
    incoming = [
        catalog_stub_for_path(path, chunks=chunks_by_path.get(path) or [], filename=Path(path).name)
        for path in sorted(replace)
    ]
    old = load_jsonl(dest)
    merged = merge_jsonl_by_path(old, incoming, replace_paths=replace)
    write_jsonl(dest, merged)
    return {"before": len(old), "incoming": len(incoming), "after": len(merged)}
