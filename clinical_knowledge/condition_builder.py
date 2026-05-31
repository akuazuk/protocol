"""Сборка JSON нозологий (формат gastro MVP) из карточки протокола и извлечённых правил."""
from __future__ import annotations

import re
from typing import Any

from .condition_registry import CONDITION_BY_ID


def _population_from_card(card: dict[str, Any]) -> str:
    pop = str(card.get("population") or "any").lower()
    if pop in ("adult", "child", "any"):
        return pop
    title = (card.get("title") or "").lower()
    if "дет" in title or "pediatr" in title:
        return "child"
    if "взросл" in title:
        return "adult"
    return "any"


def _condition_title(cid: str, card: dict[str, Any]) -> str:
    title = str(card.get("title") or "").strip()
    if title and len(title) > 8:
        return title[:500]
    cdef = CONDITION_BY_ID.get(cid)
    if cdef and cdef.text_markers:
        return cdef.text_markers[0].capitalize()
    return cid.replace("_", " ")


def _icd_from_card(card: dict[str, Any]) -> list[str]:
    icd = list(card.get("icd10_all") or card.get("icd10_primary") or [])
    return [str(x).upper() for x in icd if x][:24]


def _components_from_rules(rules: list[dict[str, Any]]) -> list[str]:
    for rule in rules:
        if rule.get("rule_type") == "diagnosis_formula":
            comps = list(rule.get("required_components") or [])
            if comps:
                return comps[:10]
    return []


def _required_exams_from_rules(rules: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rule in rules:
        if rule.get("rule_type") == "required_exam":
            exam = str(rule.get("exam") or "").strip()
            if exam:
                out.append(exam)
    return out


def build_condition_record(
    condition_id: str,
    card: dict[str, Any],
    rules: list[dict[str, Any]],
    *,
    enrichment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gastro-like condition JSON для catalog/conditions/."""
    enrich = (enrichment or {}).get("enrichment") if enrichment else {}
    if not isinstance(enrich, dict):
        enrich = {}

    components = _components_from_rules(rules) or [
        str(x).strip()
        for x in (enrich.get("diagnosis_required_components") or [])
        if str(x).strip()
    ]
    if not components:
        components = ["нозология", "клиническая форма", "степень тяжести", "осложнения"]

    approval = card.get("approval") if isinstance(card.get("approval"), dict) else {}
    icd = _icd_from_card(card) or [str(x).upper() for x in (enrich.get("icd10") or []) if x]

    record: dict[str, Any] = {
        "condition_id": condition_id,
        "condition": str(enrich.get("condition") or _condition_title(condition_id, card)),
        "icd10": icd,
        "population": _population_from_card(card),
        "protocol_reference": {
            "protocol_id": card.get("protocol_id"),
            "source_path": (card.get("source_path") or "").replace("\\", "/"),
            "approval_number": approval.get("number"),
            "approval_date": approval.get("date"),
            "specialty_ru": card.get("specialty_ru"),
        },
        "diagnosis_formula": {
            "required_components": components,
            "description_ru": f"Компоненты формулировки диагноза по КП ({condition_id}).",
        },
        "diagnostic_criteria_summary": str(enrich.get("diagnostic_criteria_summary") or "").strip(),
        "required_exams": _required_exams_from_rules(rules)
        or [str(x) for x in (enrich.get("required_exams") or []) if x][:12],
        "catalog_scope": "all_rubrics",
        "structured_from": "catalog_full_build",
    }
    red_flags = [str(x) for x in (enrich.get("red_flags") or []) if x]
    if red_flags:
        record["red_flags"] = red_flags[:12]
    block_types = ["definition", "diagnosis_formula"]
    if record.get("diagnostic_criteria_summary"):
        block_types.append("diagnostic_criteria")
    if record.get("required_exams"):
        block_types.append("exam_indications")
    record["block_types_present"] = block_types
    return record


def merge_condition_records(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Объединить две записи одной нозологии (несколько PDF)."""
    out = dict(existing)
    for key in ("condition", "diagnostic_criteria_summary"):
        if not out.get(key) and new.get(key):
            out[key] = new[key]
    icd = list(dict.fromkeys((out.get("icd10") or []) + (new.get("icd10") or [])))
    if icd:
        out["icd10"] = icd[:32]
    old_comps = (out.get("diagnosis_formula") or {}).get("required_components") or []
    new_comps = (new.get("diagnosis_formula") or {}).get("required_components") or []
    merged_comps = list(dict.fromkeys(list(old_comps) + list(new_comps)))[:12]
    out["diagnosis_formula"] = {
        **(out.get("diagnosis_formula") or {}),
        "required_components": merged_comps,
    }
    refs = out.get("protocol_references")
    if refs is None:
        refs = []
        if out.get("protocol_reference"):
            refs = [out["protocol_reference"]]
        out["protocol_references"] = refs
        out.pop("protocol_reference", None)
    new_ref = new.get("protocol_reference")
    if new_ref and isinstance(refs, list):
        paths = {r.get("source_path") for r in refs if isinstance(r, dict)}
        if new_ref.get("source_path") not in paths:
            refs.append(new_ref)
            out["protocol_references"] = refs[:48]
    exams = list(
        dict.fromkeys((out.get("required_exams") or []) + (new.get("required_exams") or []))
    )
    if exams:
        out["required_exams"] = exams[:20]
    return out


def slug_condition_from_title(title: str) -> str:
    s = re.sub(r"[^a-z0-9а-яё]+", "_", (title or "").lower()).strip("_")
    return (s[:48] or "protocol").replace("ё", "e")
