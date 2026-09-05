"""Аномалии статьи РЗ ↔ коды findings + KPI (wave 4)."""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "data" / "mo_anomalies" / "article_anomaly_codes.json"

LAB_UNUSED_CODES = {
    "B_lab_unused_in_dx",
    "B_lab_unused_in_plan",
    "B_lab_present_not_in_mo",
    "B_lab_abnormal_ignored",
    "B_lab_ordered_not_used",
}
DRUG_SAFETY_CODES = {
    "C_ddi",
    "C_nsaid_dup",
    "C_ppi_dup",
    "C_antihistamine_dup",
    "C_anticoag_dup",
    "C_statin_dup",
    "C_ace_arb_dup",
    "C_high_alert_no_dose",
    "C_rceth_off_label",
    "C_rceth_contraindication",
    "C_rceth_age_outside_label",
    "C_formulary_unknown",
}
DRUG_DDI_CODES = {"C_ddi"}
DRUG_DUP_CODES = {
    "C_nsaid_dup",
    "C_ppi_dup",
    "C_antihistamine_dup",
    "C_anticoag_dup",
    "C_statin_dup",
    "C_ace_arb_dup",
}
DRUG_LABEL_CODES = {
    "C_high_alert_no_dose",
    "C_rceth_off_label",
    "C_rceth_contraindication",
    "C_rceth_age_outside_label",
    "C_formulary_unknown",
}


@lru_cache(maxsize=1)
def load_anomaly_catalog() -> list[dict[str, Any]]:
    if not _PATH.is_file():
        return []
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    return list(data.get("anomalies") or [])


def anomaly_code_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_anomaly_catalog():
        for code in row.get("codes") or []:
            if str(code).startswith("_"):
                continue
            out[str(code)] = {
                "n": row.get("n"),
                "title_ru": row.get("title_ru"),
            }
    return out


def _codes_from_findings(findings: Iterable[Mapping[str, Any]] | None) -> set[str]:
    out: set[str] = set()
    for item in findings or []:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code") or item.get("finding_code") or "").strip()
        if code:
            out.add(code)
    return out


def classify_case_anomalies(
    findings: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    present = _codes_from_findings(findings)
    hits: list[dict[str, Any]] = []
    for row in load_anomaly_catalog():
        codes = [c for c in (row.get("codes") or []) if not str(c).startswith("_")]
        matched = sorted(present.intersection(codes))
        if not matched:
            continue
        hits.append(
            {
                "n": row.get("n"),
                "title_ru": row.get("title_ru"),
                "matched_codes": matched,
            }
        )
    return hits


def kpi_from_finding_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    total_cases: int,
) -> dict[str, Any]:
    """rows: [{mis_id, finding_code}, ...] или агрегаты с cases count."""
    by_code: Counter[str] = Counter()
    cases_unused: set[Any] = set()
    cases_drug: set[Any] = set()
    cases_ddi: set[Any] = set()
    cases_dup: set[Any] = set()
    cases_label: set[Any] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        code = str(row.get("finding_code") or row.get("code") or "")
        mid = row.get("mis_id")
        count = int(row.get("cases") or 1)
        by_code[code] += count
        if mid is None:
            continue
        if code in LAB_UNUSED_CODES:
            cases_unused.add(mid)
        if code in DRUG_SAFETY_CODES:
            cases_drug.add(mid)
        if code in DRUG_DDI_CODES:
            cases_ddi.add(mid)
        if code in DRUG_DUP_CODES:
            cases_dup.add(mid)
        if code in DRUG_LABEL_CODES:
            cases_label.add(mid)
    total = max(int(total_cases or 0), 0)

    def _pct(n: int) -> float | None:
        if total <= 0:
            return None
        return round(100.0 * n / total, 1)

    return {
        "total_cases": total,
        "unused_lab_cases": len(cases_unused),
        "unused_lab_pct": _pct(len(cases_unused)),
        "drug_safety_cases": len(cases_drug),
        "drug_safety_pct": _pct(len(cases_drug)),
        "drug_columns": {
            "interactions_cases": len(cases_ddi),
            "interactions_pct": _pct(len(cases_ddi)),
            "duplicates_cases": len(cases_dup),
            "duplicates_pct": _pct(len(cases_dup)),
            "dose_label_cases": len(cases_label),
            "dose_label_pct": _pct(len(cases_label)),
        },
        "by_code": dict(by_code.most_common(40)),
        "shadow_note_ru": (
            "KPI по unused lab и новым class-dup / formulary - в апробации "
            "(shadow), пока не включён primary."
        ),
    }
