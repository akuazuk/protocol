"""Покрытие автоизвлечения правил по каталогу протоколов."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CATALOG_COVERAGE = ROOT / "data" / "catalog" / "rules_coverage_report.json"
GASTRO_COVERAGE = ROOT / "data" / "gastro_mvp" / "rules_coverage_report.json"
SUMMARY_PATH = ROOT / "data" / "gastro_mvp" / "rules_extraction_summary.json"


def _load_protocol_paths() -> list[str]:
    from .catalog_build import catalog_source_paths

    return catalog_source_paths()


def load_rules_coverage_report() -> dict[str, Any]:
    for path in (CATALOG_COVERAGE, GASTRO_COVERAGE):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("pdfs_total"):
                    return data
            except Exception:
                pass
    paths = _load_protocol_paths()
    if paths:
        try:
            from .rules_from_corpus import infer_condition_ids_from_source_path
            from .rules_from_path import infer_path_condition

            with_rules_list: list[str] = []
            without_rules_list: list[str] = []
            by_rubric: dict[str, dict[str, int]] = {}
            for sp in paths:
                rubric = sp.replace("\\", "/").split("/")[1] if "/" in sp else "unknown"
                by_rubric.setdefault(rubric, {"pdfs": 0, "with_rules": 0})
                by_rubric[rubric]["pdfs"] += 1
                has = bool(infer_path_condition(sp) or infer_condition_ids_from_source_path(sp))
                if has:
                    with_rules_list.append(sp)
                    by_rubric[rubric]["with_rules"] += 1
                else:
                    without_rules_list.append(sp)
            total = len(paths)
            rubric_summary = {
                slug: {
                    "pdfs_total": v["pdfs"],
                    "pdfs_with_rules": v["with_rules"],
                    "coverage_pct": round(100.0 * v["with_rules"] / v["pdfs"], 1) if v["pdfs"] else 0.0,
                }
                for slug, v in sorted(by_rubric.items())
            }
            return {
                "pdfs_total": total,
                "pdfs_with_rules": len(with_rules_list),
                "pdfs_without_rules": len(without_rules_list),
                "with_rules": with_rules_list,
                "without_rules": without_rules_list,
                "by_rubric": rubric_summary,
                "scope": "all_catalog_path_heuristics",
            }
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
        "by_rubric": rep.get("by_rubric") or {},
        "scope": rep.get("scope") or "all_catalog",
        "report_path": "data/catalog/rules_coverage_report.json",
    }
