#!/usr/bin/env python3
"""Отчёт покрытия path/corpus-правил по всему каталогу protocol_cards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "catalog" / "rules_coverage_report.json"


def main() -> int:
    from clinical_knowledge.coverage import load_rules_coverage_report

    report = load_rules_coverage_report()
    if not report.get("pdfs_total"):
        from clinical_knowledge.coverage import _load_protocol_paths
        from clinical_knowledge.rules_from_corpus import infer_condition_ids_from_source_path
        from clinical_knowledge.rules_from_path import infer_path_condition

        paths = _load_protocol_paths()
        if not paths:
            print("WARN: нет protocol_cards — соберите output/registry/protocol_cards.jsonl")
            return 1

        with_rules: list[str] = []
        without_rules: list[str] = []
        by_rubric: dict[str, dict[str, int]] = {}
        for sp in paths:
            norm = sp.replace("\\", "/")
            rubric = norm.split("/")[1] if "/" in norm else "unknown"
            by_rubric.setdefault(rubric, {"pdfs": 0, "with_rules": 0})
            by_rubric[rubric]["pdfs"] += 1
            has = bool(infer_path_condition(norm) or infer_condition_ids_from_source_path(norm))
            if has:
                with_rules.append(norm)
                by_rubric[rubric]["with_rules"] += 1
            else:
                without_rules.append(norm)

        report = {
            "pdfs_total": len(paths),
            "pdfs_with_rules": len(with_rules),
            "pdfs_without_rules": len(without_rules),
            "scope": "all_catalog_path_heuristics",
            "with_rules": with_rules,
            "without_rules": without_rules,
            "by_rubric": {
                slug: {
                    "pdfs_total": v["pdfs"],
                    "pdfs_with_rules": v["with_rules"],
                    "coverage_pct": round(100.0 * v["with_rules"] / v["pdfs"], 1) if v["pdfs"] else 0.0,
                }
                for slug, v in sorted(by_rubric.items())
            },
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "pdfs_total": report.get("pdfs_total"),
                "pdfs_with_rules": report.get("pdfs_with_rules"),
                "pdfs_without_rules": report.get("pdfs_without_rules"),
                "coverage_pct": round(
                    100.0 * int(report.get("pdfs_with_rules") or 0) / max(1, int(report.get("pdfs_total") or 1)),
                    1,
                ),
                "rubrics": len(report.get("by_rubric") or {}),
                "report_path": str(OUT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
