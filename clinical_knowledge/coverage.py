"""Покрытие автоизвлечения правил по гастро-PDF."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_PATH = ROOT / "data" / "gastro_mvp" / "rules_coverage_report.json"
SUMMARY_PATH = ROOT / "data" / "gastro_mvp" / "rules_extraction_summary.json"


def load_rules_coverage_report() -> dict[str, Any]:
    if COVERAGE_PATH.is_file():
        try:
            data = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    if SUMMARY_PATH.is_file():
        try:
            s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            if isinstance(s, dict):
                return {
                    "pdfs_total": s.get("pdfs_total"),
                    "pdfs_with_rules": s.get("pdfs_with_rules"),
                    "pdfs_without_rules": (s.get("pdfs_total") or 0) - (s.get("pdfs_with_rules") or 0),
                    "total_rules": s.get("total_rules"),
                    "rules_by_condition": s.get("rules_by_condition"),
                }
        except Exception:
            pass
    return {}


def coverage_status_payload() -> dict[str, Any]:
    rep = load_rules_coverage_report()
    total = int(rep.get("pdfs_total") or 0)
    with_rules = int(rep.get("pdfs_with_rules") or 0)
    without = int(rep.get("pdfs_without_rules") or max(0, total - with_rules))
    pct = round(100.0 * with_rules / total, 1) if total else 0.0
    return {
        "pdfs_total": total,
        "pdfs_with_rules": with_rules,
        "pdfs_without_rules": without,
        "coverage_pct": pct,
        "total_auto_rules": int(rep.get("total_rules") or 0),
        "rules_by_condition": rep.get("rules_by_condition") or {},
        "report_path": "data/gastro_mvp/rules_coverage_report.json",
    }
