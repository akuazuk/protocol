"""Детерминированные правила по пути PDF (острые/хирургические КП без блока «формулировка диагноза»)."""
from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

# needles в lower(source_path) → condition_id, компоненты диагноза
PATH_CONDITION_TEMPLATES: list[dict[str, Any]] = [
    {
        "needles": ("аппендицит",),
        "condition_id": "acute_appendicitis",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("панкреатит",),
        "condition_id": "acute_pancreatitis",
        "required_components": ["нозология", "форма", "тяжесть"],
    },
    {
        "needles": ("холецистит",),
        "condition_id": "acute_cholecystitis",
        "required_components": ["нозология", "форма", "тяжесть"],
    },
    {
        "needles": ("непроходимост",),
        "condition_id": "intestinal_obstruction",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("инвагинац",),
        "condition_id": "intussusception",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("грыж",),
        "condition_id": "incarcerated_hernia",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("инородн",),
        "condition_id": "foreign_body_gi",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("кровотеч",),
        "condition_id": "gi_bleeding",
        "required_components": ["нозология", "источник", "тяжесть"],
    },
    {
        "needles": ("перфоратив",),
        "condition_id": "perforated_peptic_ulcer",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("травм", "живот"),
        "condition_id": "abdominal_trauma",
        "required_components": ["нозология", "механизм", "тяжесть"],
        "match_all": True,
    },
    {
        "needles": ("дефекац", "эвакуатор"),
        "condition_id": "defecation_disorder",
        "required_components": ["нозология", "форма", "степень"],
    },
    {
        "needles": ("общехирург",),
        "condition_id": "pediatric_general_surgery",
        "required_components": ["нозология", "форма"],
    },
    {
        "needles": ("целиак",),
        "condition_id": "celiac",
        "required_components": ["нозология", "клиническая форма", "период"],
    },
    {
        "needles": ("прямой_кишки", "доброкач"),
        "condition_id": "rectal_neoplasm",
        "required_components": ["нозология", "локализация", "стадия"],
        "match_all": True,
    },
]


def infer_path_condition(source_path: str) -> tuple[str, list[str]] | None:
    low = (source_path or "").lower().replace("\\", "/")
    for tpl in PATH_CONDITION_TEMPLATES:
        needles = tpl.get("needles") or ()
        if tpl.get("match_all"):
            if all(n in low for n in needles):
                return str(tpl["condition_id"]), list(tpl["required_components"])
        elif any(n in low for n in needles):
            return str(tpl["condition_id"]), list(tpl["required_components"])
    return None


def extract_path_rules(
    source_path: str,
    *,
    protocol_id: str,
    rule_id_prefix: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Правила по шаблону пути PDF."""
    inferred = infer_path_condition(source_path)
    if not inferred:
        return {}
    cid, components = inferred
    prefix = (rule_id_prefix + "_") if rule_id_prefix else ""
    pdf_hash = sha256(source_path.encode()).hexdigest()[:8]
    rule = {
        "rule_id": f"{prefix}path_{cid}_diagnosis_formula",
        "rule_type": "diagnosis_formula",
        "required_components": components,
        "severity": "warning",
        "description_ru": f"Шаблон по пути КП: полнота диагноза ({cid}).",
        "source": {
            "protocol_id": protocol_id,
            "source_path": source_path.replace("\\", "/"),
            "path_inferred": True,
        },
        "auto_extracted": True,
        "extraction_method": "path_template",
    }
    return {cid: [rule]}


def path_rules_for_uncovered(
    source_paths: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Собрать path-правила для списка PDF без regex-извлечения."""
    merged: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for sp in source_paths:
        pdf_hash = sha256(sp.encode()).hexdigest()[:8]
        protocol_id = f"gastro_{pdf_hash}"
        for cid, rules in extract_path_rules(
            sp, protocol_id=protocol_id, rule_id_prefix=pdf_hash
        ).items():
            for rule in rules:
                rid = str(rule.get("rule_id") or "")
                if rid in seen:
                    continue
                seen.add(rid)
                merged.setdefault(cid, []).append(rule)
    return merged
