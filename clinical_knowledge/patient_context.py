"""Контекст пациента из L1 / КЗ для B2C фильтров и индексов."""
from __future__ import annotations

import re
from typing import Any

from .age_sex_resolver import adult_or_child


def _doc_from_l1(l1_result: dict[str, Any]) -> dict[str, Any]:
    sa = l1_result.get("structured_analysis") if isinstance(l1_result.get("structured_analysis"), dict) else {}
    doc = sa.get("document") if isinstance(sa.get("document"), dict) else {}
    return doc


def _infer_specialty(doc: dict[str, Any], kz_text: str = "") -> str | None:
    blob = (kz_text or "").lower()
    for key in ("doctor_specialty", "specialty", "specialty_slug"):
        val = str(doc.get(key) or "").strip().lower()
        if val:
            if "nevrolog" in val or "невролог" in val:
                return "neurology"
            if "flebolog" in val or "флеболог" in val:
                return "phlebology"
            if "dermat" in val or "дермат" in val:
                return "dermatology"
            return val[:48]
    if "невролог" in blob:
        return "neurology"
    if "флеболог" in blob:
        return "phlebology"
    if "дерматовенеролог" in blob or "дерматолог" in blob:
        return "dermatology"
    if "l93" in blob.replace(" ", "") or "волчан" in blob:
        return "dermatology"
    return None


def _icd_codes(doc: dict[str, Any], kz_text: str = "") -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for d in doc.get("diagnoses") or []:
        if not isinstance(d, dict):
            continue
        code = str(d.get("icd10_code") or "").strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    if not out and kz_text:
        for m in re.finditer(r"\b([A-Z]\d{2}(?:\.\d)?)\b", kz_text.upper()):
            c = m.group(1)
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:8]


def extract_patient_context(
    l1_result: dict[str, Any],
    *,
    kz_text: str = "",
    demographics_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """age, age_group, sex, specialty, icd10_codes, care_type."""
    doc = _doc_from_l1(l1_result)
    patient = doc.get("patient") if isinstance(doc.get("patient"), dict) else {}
    meta = demographics_meta or {}

    age_years = patient.get("age_years") or meta.get("age_years")
    if age_years is None and kz_text:
        m = re.search(r"(?:дата рождения|г\.?\s*р\.?)[^\d]{0,20}(\d{4})", kz_text.lower())
        if m:
            from datetime import date

            birth = int(m.group(1))
            age_years = max(0, date.today().year - birth)

    age_group = patient.get("adult_or_child") or patient.get("age_group")
    if not age_group or age_group == "unknown":
        age_group = adult_or_child(int(age_years) if age_years is not None else None)

    sex = patient.get("sex") or meta.get("sex")
    if not sex and kz_text:
        low = kz_text.lower()
        if re.search(r"\b(женск|female|ж\.)\b", low):
            sex = "female"
        elif re.search(r"\b(мужск|male|м\.)\b", low):
            sex = "male"

    return {
        "age": int(age_years) if age_years is not None else None,
        "age_group": age_group if age_group in ("adult", "child", "unknown") else "unknown",
        "sex": sex,
        "specialty": _infer_specialty(doc, kz_text),
        "icd10_codes": _icd_codes(doc, kz_text),
        "document_type": "consultation_conclusion",
        "care_type": "outpatient",
    }
