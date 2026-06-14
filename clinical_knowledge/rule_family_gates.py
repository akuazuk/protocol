"""Семейные gates для rule_checker: ICD + specialty → применимость нозологии."""
from __future__ import annotations

import re
from typing import Any


def is_oncology_icd(code: str) -> bool:
    """ЗНО-контекст: C*, D00-D09 in situ, D37-D48 uncertain; не D10-D36 (доброкач.)."""
    c = (code or "").upper().strip()
    if not c:
        return False
    if c.startswith("C"):
        return True
    if c.startswith("D") and len(c) >= 3 and c[1:3].isdigit():
        n = int(c[1:3])
        if 0 <= n <= 9:
            return True
        if 10 <= n <= 36:
            return False
        if 37 <= n <= 48:
            return True
    return False


def expand_specialty_slugs_for_icd(
    slugs: set[str] | list[str] | None,
    icd_codes: list[str] | None,
) -> set[str]:
    """Доп. рубрики каталога по МКБ (ОРВИ → пульмонология/инфекции; D25 → акушерство)."""
    out = {s.strip() for s in (slugs or []) if s and str(s).strip()}
    for raw in icd_codes or []:
        c = str(raw or "").upper().strip()
        if not c:
            continue
        root = c[:3] if len(c) >= 3 else c
        if root.startswith("J06") or root in {"J00", "J02", "J03", "J04", "J05", "J11"}:
            out.update(
                {
                    "pulmonologiya-ftiziatriya",
                    "infektsionnye-zabolevaniya",
                    "terapiya",
                }
            )
        if root.startswith("D25") or root.startswith(("N80", "N81", "N82", "N83", "N84", "N85", "N86", "N87", "N88", "N89", "N90", "N91", "N92", "N93", "N94", "N95", "N97")):
            out.add("akusherstvo-ginekologiya")
        if root.startswith(("E03", "E04", "E05", "E06")):
            out.add("endokrinologiya-narusheniya-obmena-veshchestv")
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _text_blob(consult_facts: dict[str, Any]) -> str:
    cons = consult_facts.get("consultation") or {}
    parts = [
        cons.get("diagnosis_text") or "",
        " ".join(cons.get("complaints") or []),
        cons.get("text_sample") or "",
    ]
    return _norm(" ".join(parts))


def condition_family_applies(condition_id: str, consult_facts: dict[str, Any]) -> bool | None:
    """None — использовать стандартную логику rule_checker."""
    cons = consult_facts.get("consultation") or {}
    icd_list = [str(x).upper() for x in (cons.get("icd10") or []) if x]
    diag = _norm(str(cons.get("diagnosis_text") or ""))

    if condition_id == "acute_bronchitis":
        has_bronchitis_icd = any(c.startswith(("J20", "J21")) for c in icd_list)
        has_bronchitis_text = "бронхит" in diag
        urti_only = bool(icd_list) and all(
            c.startswith(("J00", "J01", "J02", "J03", "J04", "J05", "J06")) for c in icd_list
        )
        if urti_only and not has_bronchitis_icd and not has_bronchitis_text:
            return False

    if condition_id == "obesity":
        if icd_list and all(c.startswith(("E03", "E04", "E05", "E06")) for c in icd_list):
            return False
        if "эутиреоз" in diag or "euthyre" in diag:
            return False

    if condition_id in ("tuberculosis", "pneumonia", "bronchial_asthma"):
        urti_only = bool(icd_list) and all(
            c.startswith(("J00", "J01", "J02", "J03", "J04", "J05", "J06")) for c in icd_list
        )
        if urti_only:
            return False

    if condition_id == "dermatitis":
        if icd_list and not any(c.startswith("L") for c in icd_list):
            blob = _text_blob(consult_facts)
            if not any(m in blob for m in ("дерматит", "экзем", "кож", "сып")):
                return False

    return None
