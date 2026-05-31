#!/usr/bin/env python3
"""Отчёт покрытия path/corpus-правил по всему каталогу protocol_cards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "gastro_mvp" / "rules_coverage_report.json"


def main() -> int:
    from clinical_knowledge.coverage import _load_protocol_paths
    from clinical_knowledge.rules_from_corpus import infer_condition_ids_from_source_path
    from clinical_knowledge.rules_from_path import infer_path_condition

    paths = _load_protocol_paths()
    if not paths:
        print("WARN: нет protocol_cards — соберите output/registry/protocol_cards.jsonl")
        return 1

    with_rules: list[str] = []
    without_rules: list[str] = []
    per_pdf: dict[str, dict] = {}

    for sp in paths:
        norm = sp.replace("\\", "/")
        path_hit = infer_path_condition(norm)
        corpus_ids = infer_condition_ids_from_source_path(norm)
        has = bool(path_hit or corpus_ids)
        per_pdf[norm] = {
            "path_condition": path_hit[0] if path_hit else None,
            "corpus_conditions": corpus_ids,
            "rules": 1 if has else 0,
        }
        if has:
            with_rules.append(norm)
        else:
            without_rules.append(norm)

    report = {
        "pdfs_total": len(paths),
        "pdfs_with_rules": len(with_rules),
        "pdfs_without_rules": len(without_rules),
        "scope": "all_catalog",
        "with_rules": with_rules,
        "without_rules": without_rules,
        "per_pdf": per_pdf,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pdfs_total": report["pdfs_total"],
                "pdfs_with_rules": report["pdfs_with_rules"],
                "pdfs_without_rules": report["pdfs_without_rules"],
                "coverage_pct": round(100.0 * len(with_rules) / len(paths), 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
