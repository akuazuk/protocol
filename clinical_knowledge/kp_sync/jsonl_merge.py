"""Merge jsonl/json корпуса по source_path: старые строки остаются, changed заменяются."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _path_key(row: dict[str, Any]) -> str:
    for k in ("source_path", "path", "relative_path"):
        val = row.get(k)
        if val:
            return str(val).replace("\\", "/")
    return ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_jsonl_by_path(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    replace_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Заменяет все строки, чей path в incoming (или в replace_paths). Остальные keep."""
    incoming_paths = {_path_key(r) for r in incoming if _path_key(r)}
    drop = set(replace_paths or set()) | incoming_paths
    kept = [r for r in existing if _path_key(r) not in drop]
    return kept + list(incoming)


def merge_tables_json(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    replace_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    return merge_jsonl_by_path(existing, incoming, replace_paths=replace_paths)


def merge_jsonl_files(
    dest: Path,
    incoming: list[dict[str, Any]],
    *,
    replace_paths: set[str] | None = None,
) -> dict[str, int]:
    old = load_jsonl(dest)
    merged = merge_jsonl_by_path(old, incoming, replace_paths=replace_paths)
    write_jsonl(dest, merged)
    return {
        "before": len(old),
        "incoming": len(incoming),
        "after": len(merged),
        "replaced_paths": len(replace_paths or set()) or len(
            {_path_key(r) for r in incoming if _path_key(r)}
        ),
    }
