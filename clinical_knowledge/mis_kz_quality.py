"""Загрузка агрегатов L1-анализа mis_protocol для кабинета методиста."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _candidate_summary_paths(month: str | None = None) -> list[Path]:
    month = (month or "").strip() or "2026-07"
    name = f"kz_l1_{month}_summary.json"
    env = (os.environ.get("MIS_KZ_SUMMARY_PATH") or "").strip()
    out: list[Path] = []
    if env:
        out.append(Path(env))
    out.append(Path("/var/data/mis_protocol") / name)
    out.append(ROOT / "data" / "mis_protocol" / name)
    # any other months on disk / in repo
    for d in (Path("/var/data/mis_protocol"), ROOT / "data" / "mis_protocol"):
        if d.is_dir():
            out.extend(sorted(d.glob("kz_l1_*_summary.json"), reverse=True))
    return out


def load_mis_kz_summary(*, month: str | None = None) -> dict[str, Any] | None:
    seen: set[str] = set()
    for path in _candidate_summary_paths(month):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data["_source_path"] = str(path)
        return data
    return None


def build_mis_kz_quality_view(*, month: str | None = None) -> dict[str, Any]:
    summary = load_mis_kz_summary(month=month)
    if summary is None:
        return {
            "ok": False,
            "available": False,
            "error": "summary_not_found",
            "hint_ru": (
                "Нет kz_l1_*_summary.json. Запустите "
                "scripts/run_mis_protocol_l1_batch.py на Render и положите summary "
                "в /var/data/mis_protocol/ или data/mis_protocol/."
            ),
            "month": month or "2026-07",
        }
    doctors = summary.get("doctors") or []
    return {
        "ok": True,
        "available": True,
        "month": summary.get("month"),
        "tier": summary.get("tier") or "L1",
        "generated_at": summary.get("generated_at"),
        "source_path": summary.get("_source_path"),
        "n_cases": summary.get("n_cases"),
        "n_ok": summary.get("n_ok"),
        "n_errors": summary.get("n_errors"),
        "avg_overall_pct": summary.get("avg_overall_pct"),
        "median_overall_pct": summary.get("median_overall_pct"),
        "score_histogram": summary.get("score_histogram") or {},
        "status_counts": summary.get("status_counts") or {},
        "block_avg": summary.get("block_avg") or {},
        "doctors": doctors,
        "specialties": summary.get("specialties") or [],
        "filials": summary.get("filials") or [],
        "top_doctors": summary.get("top_doctors") or doctors[:15],
        "bottom_doctors": summary.get("bottom_doctors") or [],
        "worst_visits": summary.get("worst_visits") or [],
        "worst_visits_meta": summary.get("worst_visits_meta") or {},
        "notes": summary.get("notes") or [],
        "doctors_n": len(doctors),
    }
