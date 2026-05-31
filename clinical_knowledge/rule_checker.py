"""Детерминированная проверка КЗ по правилам нозологий (MVP гастро)."""
from __future__ import annotations

import re
from typing import Any

from .loader import load_conditions, load_rules_by_condition
from .rule_filter import filter_rules_for_matched_protocols, matched_source_paths


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


def _check_diagnosis_components(
    diagnosis_text: str,
    required: list[str],
) -> tuple[list[str], list[str]]:
    low = _norm(diagnosis_text)
    present: list[str] = []
    missing: list[str] = []
    markers = {
        "нозология": ("гэрб", "рефлюкс", "гастрит", "язв", "k21", "k29", "k25", "k26", "диспепс", "колит", "крон", "целиак", "k50", "k51", "k90", "k85", "k35", "панкреат", "аппендицит"),
        "клиническая форма": ("форма", "неэрозив", "эрозив", "атроф", "поверхност", "катаральн", "флегмон"),
        "форма": ("форма", "лёгк", "легк", "умерен", "тяжел", "тяжёл", "катаральн", "флегмон", "гангрен"),
        "степень тяжести": ("степен", "лёгк", "легк", "средн", "тяжел", "тяжёл"),
        "фаза": ("фаз", "обострен", "ремисс", "хроническ", "остр"),
        "осложнения": ("осложнен", "кровотеч", "стеноз", "перфора", "без ослож", "перитонит", "абсцесс"),
        "источник": ("источник", "язв", "варикоз", "эроз", "диvert", "диверт"),
        "механизм": ("механизм", "удар", "паден", "дтп", "нож", "огнестр"),
        "локализация": ("локал", "антрал", "луковиц", "желудк", "двенадцат", "l1", "l2", "l3", "l4", "илеальн", "толстокиш"),
        "h.pylori": ("нр", "hp", "helicobacter", "хеликобактер"),
        "этиологический фактор": ("этиолог", "нр", "нпвп", "стресс"),
        "активность": ("активност", "hp 3", "воспален", "обострен", "ремисс"),
        "атрофия": ("атроф", "olga"),
        "метаплазия": ("метаплаз", "olgim"),
        "острый или хронический характер": ("остр", "хроническ"),
        "протяженность": ("протяжен", "дистальн", "тотальн", "леоколит", "панколит", "проктит", "левая"),
        "протяженность поражения кишечника": ("проктит", "панколит", "левая", "тотальн", "левосторон"),
        "фазу течения – обострение или ремиссия": ("обострен", "ремисс", "фаз"),
        "характер течения": ("остр", "хроническ", "рецидив", "непрерывн"),
        "тяжесть текущего обострения в соответствии с пиаяк": ("тяжел", "тяжёл", "лёгк", "легк", "умерен", "пиаяк"),
        "тяжесть": ("тяжел", "тяжёл", "среднетяж", "лёгк", "легк", "умерен"),
        "диагноз в соответствии с парижской педиатрической классификацией бк": ("l1", "l2", "l3", "a1", "b1", "b2", "b3", "локализац", "илеальн", "толстокиш"),
        "вариант течения": ("вариант", "течени", "рекуррент"),
        "период": ("период", "ремисс", "обострен"),
        "гистологическая стадия": ("гистолог", "стади", "marsh"),
    }
    for comp in required:
        key = comp.lower()
        found = False
        if key in low:
            found = True
        else:
            for m in markers.get(key, ()):
                if m in low:
                    found = True
                    break
        if found:
            present.append(comp)
        else:
            missing.append(comp)
    return present, missing


def _check_symptom_duration_frequency(text: str, criterion: dict[str, Any]) -> bool:
    low = _norm(text)
    sym = _norm(str(criterion.get("symptom") or ""))
    if sym:
        tokens = [t for t in re.split(r"[\s/или]+", sym) if len(t) > 3]
        if tokens and not any(t in low for t in tokens):
            return False
    dur = str(criterion.get("duration") or "")
    if ">=" in dur and "месяц" in dur:
        m = re.search(r">=\s*(\d+)", dur)
        if m:
            months = int(m.group(1))
            if not re.search(rf"{months}\s*(?:мес|месяц)", low) and "месяц" not in low:
                if months >= 6 and not any(x in low for x in ("6 мес", "6 месяц", "полгода", "давно", "хронич")):
                    return False
    freq = str(criterion.get("frequency") or "")
    if ">=" in freq and "недел" in freq:
        if not any(x in low for x in ("2 раз", "два раз", "еженед", "част", "ежеднев")):
            return False
    finding = _norm(str(criterion.get("finding") or ""))
    if finding and finding not in low:
        method = _norm(str(criterion.get("method") or ""))
        if method and method not in low:
            return False
    return True


def _run_rule(rule: dict[str, Any], consult_facts: dict[str, Any]) -> dict[str, Any]:
    rule_type = rule.get("rule_type") or ""
    text = _text_blob(consult_facts)
    diagnosis = (consult_facts.get("consultation") or {}).get("diagnosis_text") or ""
    ctx = consult_facts.get("patient_context") or {}

    finding: dict[str, Any] = {
        "rule_id": rule.get("rule_id"),
        "rule_type": rule_type,
        "severity": rule.get("severity") or "warning",
        "passed": True,
        "message_ru": "",
        "missing": [],
        "source": rule.get("source"),
    }

    if rule_type == "population_mismatch":
        expected = (rule.get("expected_population") or "").lower()
        actual = (ctx.get("adult_or_child") or "").lower()
        if expected and actual and expected != actual:
            finding["passed"] = False
            finding["severity"] = "critical"
            finding["message_ru"] = (
                f"Протокол для {expected}, в КЗ указана аудитория {actual}."
            )
        return finding

    if rule_type == "diagnosis_formula":
        required = list(rule.get("required_components") or [])
        present, missing = _check_diagnosis_components(diagnosis, required)
        finding["present"] = present
        finding["missing"] = missing
        if missing:
            finding["passed"] = False
            finding["message_ru"] = (
                "В формулировке диагноза не хватает компонентов: "
                + ", ".join(missing)
            )
        return finding

    if rule_type == "diagnostic_criterion":
        logic = (rule.get("logic") or "any_of").lower()
        if logic == "reference_only":
            finding["passed"] = True
            finding["severity"] = "info"
            finding["message_ru"] = rule.get("description_ru") or ""
            return finding
        criteria = list(rule.get("criteria") or [])
        hits = [c for c in criteria if _check_symptom_duration_frequency(text + " " + diagnosis, c)]
        if logic == "any_of":
            finding["passed"] = bool(hits)
            if not hits:
                finding["message_ru"] = (
                    "Не найдено достаточных диагностических критериев в тексте КЗ "
                    f"({rule.get('description_ru') or rule.get('rule_id')})."
                )
        else:
            finding["passed"] = len(hits) == len(criteria) and bool(criteria)
            if not finding["passed"]:
                finding["message_ru"] = "Не все диагностические критерии подтверждены текстом КЗ."
        finding["criteria_matched"] = len(hits)
        return finding

    if rule_type == "required_exam":
        exam = _norm(str(rule.get("exam") or ""))
        if exam and exam not in _norm(text):
            finding["passed"] = False
            finding["message_ru"] = f"В КЗ не упомянуто обязательное обследование: {rule.get('exam')}."
        return finding

    if rule_type == "keyword_presence":
        kw = _norm(str(rule.get("keyword") or ""))
        if kw and kw not in _norm(text):
            finding["passed"] = False
            finding["message_ru"] = rule.get("message_ru") or f"Ожидалось упоминание: {rule.get('keyword')}."
        return finding

    return finding


def run_rule_checker(
    consult_facts: dict[str, Any],
    *,
    condition_ids: list[str] | None = None,
    matched_protocols: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Проверка КЗ по правилам MVP; возвращает findings и сводку."""
    conditions = load_conditions()
    rules_map = load_rules_by_condition()
    cons = consult_facts.get("consultation") or {}
    hints = list(cons.get("conditions_hint") or [])

    if condition_ids:
        target = condition_ids
    elif hints:
        target = hints
    else:
        target = list(conditions.keys())

    all_findings: list[dict[str, Any]] = []
    checked_conditions: list[str] = []

    for cid in target:
        cond = conditions.get(cid)
        rules = rules_map.get(cid) or []
        if not cond and not rules:
            continue
        checked_conditions.append(cid)
        ctx = consult_facts.get("patient_context") or {}
        if cond:
            pop = cond.get("population")
            if pop and ctx.get("adult_or_child") and pop != ctx.get("adult_or_child"):
                all_findings.append(
                    {
                        "rule_id": f"{cid}_population_guard",
                        "rule_type": "population_mismatch",
                        "severity": "critical",
                        "passed": False,
                        "message_ru": (
                            f"Нозология «{cond.get('condition')}» — протокол для "
                            f"{pop}, в КЗ аудитория {ctx.get('adult_or_child')}."
                        ),
                        "source": cond.get("protocol_reference"),
                    }
                )
        for rule in filter_rules_for_matched_protocols(rules, matched_protocols):
            all_findings.append(_run_rule(rule, consult_facts))

    failed = [f for f in all_findings if not f.get("passed")]
    critical = [f for f in failed if f.get("severity") == "critical"]

    compliance_pct = None
    if all_findings:
        passed_n = sum(1 for f in all_findings if f.get("passed"))
        compliance_pct = round(100.0 * passed_n / len(all_findings), 1)

    return {
        "checked_conditions": checked_conditions,
        "findings": all_findings,
        "failed_count": len(failed),
        "critical_count": len(critical),
        "missing_required_items": [f.get("message_ru") for f in failed if f.get("message_ru")],
        "rules_compliance_pct": compliance_pct,
        "method": "deterministic_rules_v2",
        "matched_source_paths": sorted(matched_source_paths(matched_protocols)),
        "rules_filtered_by_protocol": bool(matched_protocols),
    }
