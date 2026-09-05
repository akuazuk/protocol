"""Soft formulary / known-INN check (wave 3). Shadow by default."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from clinical_knowledge.drug_normalizer import extract_drugs

ENGINE = "mo_formulary_v1"
_SOURCE = "mo_formulary_seed_v1"
CODE_FORMULARY_UNKNOWN = "C_formulary_unknown"
_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "data" / "drug_safety" / "formulary_seed.json"


def formulary_findings_enabled() -> bool:
    raw = (os.environ.get("MO_FORMULARY_FINDINGS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def formulary_primary_enabled() -> bool:
    raw = (os.environ.get("MO_FORMULARY_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def known_inns() -> set[str]:
    if not _PATH.is_file():
        return set()
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    return {
        str(x).lower().replace("ё", "е").strip()
        for x in (data.get("known_inns") or [])
        if x
    }


def formulary_findings(case: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not formulary_findings_enabled() or not isinstance(case, Mapping):
        return []
    treatment = str(case.get("treatment_recommendations") or "")
    if not treatment.strip():
        return []
    drugs = extract_drugs(treatment) or []
    known = known_inns()
    if not known:
        return []
    unknown: list[str] = []
    for drug in drugs:
        inn = str(drug.get("inn") or "").lower().replace("ё", "е").strip()
        surface = str(drug.get("surface") or "").strip()
        if not inn:
            continue
        if float(drug.get("confidence") or 0) < 0.86:
            continue
        if inn not in known:
            unknown.append(surface or inn)
    if not unknown:
        return []
    shadow = not formulary_primary_enabled()
    bits = ", ".join(unknown[:6])
    return [
        {
            "code": CODE_FORMULARY_UNKNOWN,
            "axis": "safety",
            "severity": "P3",
            "severity_label_ru": "Оформление",
            "passed": False,
            "title_ru": "Препарат не найден в локальном seed формуляра",
            "detail_ru": (
                f"Не сопоставлены с seed реестра: {bits}."
                + (" Черновик, не входит в оценку." if shadow else "")
            ),
            "evidence": treatment[:400],
            "source_ref": _SOURCE,
            "needs_human": True,
            "shadow": shadow,
            "is_shadow": shadow,
            "engine": ENGINE,
        }
    ]
