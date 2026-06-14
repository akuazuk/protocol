#!/usr/bin/env python3
"""Сборка data/protocol_catalog.jsonl и обновление index.csv (МКБ + audience)."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_catalog_module():
    path = ROOT / "clinical_knowledge" / "protocol_catalog.py"
    spec = importlib.util.spec_from_file_location("protocol_catalog", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def update_index_csv(rows: list[dict]) -> None:
    idx_path = ROOT / "index.csv"
    by_path = {r["path"]: r for r in rows}
    if not idx_path.is_file():
        return
    with idx_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for col in (
            "icd10_primary",
            "icd_count",
            "audience_confidence",
            "protocol_kind",
            "scope_label_ru",
        ):
            if col not in fieldnames:
                fieldnames.append(col)
        out_rows = []
        for row in reader:
            path = (row.get("relative_path") or "").replace("\\", "/")
            ent = by_path.get(path) or {}
            if ent.get("audience") and (not row.get("audience") or row.get("audience") == "any"):
                row["audience"] = ent["audience"]
            row["icd10_primary"] = "|".join(ent.get("icd10_primary") or [])
            row["icd_count"] = str(ent.get("icd_count") or 0)
            row["audience_confidence"] = ent.get("audience_source") or ""
            row["protocol_kind"] = ent.get("protocol_kind") or ""
            row["scope_label_ru"] = ent.get("scope_label_ru") or ""
            out_rows.append(row)
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)


def main() -> int:
    pc = _load_catalog_module()
    rows = pc.build_protocol_catalog(write=True)
    update_index_csv(rows)
    stats = pc.catalog_stats()
    print(json.dumps({"catalog": str(pc.CATALOG_PATH), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
