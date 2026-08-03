#!/usr/bin/env python3
"""Аудит покрытия МКБ и audience для всех PDF протоколов."""
from __future__ import annotations

import argparse
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


def run_audit(*, strict: bool = False) -> dict:
    pc = _load_catalog_module()
    cat = pc.load_protocol_catalog()
    if not cat:
        pc.build_protocol_catalog(write=True)
        cat = pc.load_protocol_catalog()

    gaps_icd: list[str] = []
    gaps_aud: list[str] = []
    invalid_icd: list[dict] = []
    general_marked: list[dict] = []

    valid = pc._valid_icd_codes()

    for path, row in sorted(cat.items()):
        if not row.get("icd10_all") and not row.get("general_scope"):
            gaps_icd.append(path)
        if row.get("general_scope"):
            general_marked.append(
                {
                    "path": path,
                    "kind": row.get("protocol_kind"),
                    "label": row.get("scope_label_ru"),
                }
            )
        aud = row.get("audience") or "any"
        if aud == "any":
            gaps_aud.append(path)
        for code in row.get("icd10_all") or []:
            if valid and code not in valid:
                invalid_icd.append({"path": path, "code": code})

    stats = pc.catalog_stats()
    report = {
        **stats,
        "gaps_icd_count": len(gaps_icd),
        "gaps_audience_any_count": len(gaps_aud),
        "invalid_icd_count": len(invalid_icd),
        "gaps_icd_sample": gaps_icd[:15],
        "gaps_aud_sample": gaps_aud[:15],
        "general_scope_sample": general_marked[:20],
        "general_scope_count": len(general_marked),
    }

    if strict:
        ok = stats["total"] >= 400 and stats["with_icd"] >= stats["total"] * 0.85
        report["strict_ok"] = ok
        if not ok:
            report["strict_error"] = (
                f"Coverage too low: {stats['with_icd']}/{stats['total']} with ICD"
            )

    gaps_md = ROOT / "data" / "audit" / "protocol_gaps.md"
    gaps_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Protocol catalog gaps",
        "",
        f"- Total PDFs: **{stats['total']}**",
        f"- With ICD: **{stats['with_icd']}**",
        f"- Without ICD: **{stats['without_icd']}**",
        f"- Audience explicit (not any): **{stats['with_explicit_audience']}**",
        f"- Marked general_scope: **{stats.get('general_scope_count', 0)}**",
        "",
        "## General / organizational protocols (sample)",
        "",
    ]
    for item in report.get("general_scope_sample") or []:
        lines.append(f"- `{item.get('label')}` - `{item.get('path')}`")
    lines.extend(["", "## PDF without ICD and not general (gaps)", ""])
    for p in gaps_icd[:30]:
        lines.append(f"- `{p}`")
    lines.extend(["", "## PDF with audience=any (sample)", ""])
    for p in gaps_aud[:30]:
        lines.append(f"- `{p}`")
    gaps_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["gaps_report"] = str(gaps_md)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = run_audit(strict=args.strict)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report.get("strict_ok", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
