"""Drug–disease shadow stub (wave 3). Без полного DDSI - узкий seed."""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from clinical_knowledge.drug_normalizer import extract_drugs

ENGINE = "mo_drug_disease_v1"
CODE = "C_drug_disease_mismatch"

# Узкий seed: препарат → ожидаемые подстроки диагноза (хотя бы одна).
_SEED: dict[str, tuple[str, ...]] = {
    "метформин": ("диабет", "e11", "e10", "глюкоз", "гипергликем"),
    "инсулин": ("диабет", "e11", "e10", "гипергликем"),
    "левотироксин": ("гипотиреоз", "e03", "e01", "щитовид"),
    "варфарин": ("фибрилляц", "тромбоз", "i48", "эмбол", "протез"),
}


def drug_disease_enabled() -> bool:
    raw = (os.environ.get("MO_DRUG_DISEASE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def drug_disease_findings(case: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not drug_disease_enabled() or not isinstance(case, Mapping):
        return []
    treatment = str(case.get("treatment_recommendations") or "")
    dx = " ".join(
        str(case.get(k) or "")
        for k in ("clinical_diagnosis", "diagnosis_main_text", "diagnosis")
    ).lower().replace("ё", "е")
    if not treatment.strip() or not dx.strip():
        return []
    drugs = extract_drugs(treatment) or []
    bad: list[str] = []
    for drug in drugs:
        inn = str(drug.get("inn") or "").lower().replace("ё", "е")
        surface = str(drug.get("surface") or inn)
        needles = _SEED.get(inn)
        if not needles:
            continue
        if any(n in dx for n in needles):
            continue
        bad.append(surface or inn)
    if not bad:
        return []
    bits = ", ".join(bad[:5])
    return [
        {
            "code": CODE,
            "axis": "safety",
            "severity": "P2",
            "severity_label_ru": "Умеренно",
            "passed": False,
            "title_ru": "Назначение слабо согласуется с диагнозом",
            "detail_ru": (
                f"Препараты без опоры в диагнозе (seed): {bits}. "
                "Черновик, не входит в оценку."
            ),
            "evidence": treatment[:400],
            "source_ref": ENGINE,
            "needs_human": True,
            "shadow": True,
            "is_shadow": True,
            "engine": ENGINE,
        }
    ]
