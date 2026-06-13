"""Детерминированная проверка КЗ по правилам нозологий (весь каталог)."""
from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from .condition_registry import CONDITION_BY_ID, infer_conditions_hints
from .rule_labels_ru import localize_message_ru, population_ru, rule_title_ru
from .loader import load_conditions, load_rules_by_condition
from .rule_filter import filter_rules_for_matched_protocols, matched_source_paths
from .rule_model import legacy_rule_to_protocol_rule, rule_applicable_to_patient


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


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _icd_lists_overlap(cond_icd: list[Any], consult_icd: list[str]) -> bool:
    cond_roots = {_icd_root(str(x)) for x in cond_icd if x}
    cons_roots = {_icd_root(str(x)) for x in consult_icd if x}
    if not cond_roots or not cons_roots:
        return True
    if cond_roots & cons_roots:
        return True
    for c in consult_icd:
        cu = str(c).upper()
        for cc in cond_icd:
            cuu = str(cc).upper()
            if cu.startswith(_icd_root(cuu)) or cuu.startswith(_icd_root(cu)):
                return True
    return False


def _condition_applies_to_consult(
    cid: str,
    conditions: dict[str, dict[str, Any]],
    consult_facts: dict[str, Any],
) -> bool:
    """Отсекает чужие нозологии (ИМ при I80.1) и несовпадение возраста."""
    ctx = consult_facts.get("patient_context") or {}
    cons = consult_facts.get("consultation") or {}
    icd_list = [str(x).upper() for x in (cons.get("icd10") or []) if x]
    meta = conditions.get(cid) or _condition_meta(cid, conditions) or {}
    pop = str(meta.get("population") or "any").lower()
    aud = str(ctx.get("adult_or_child") or "").lower()
    if pop not in ("any", "", "unknown") and aud and pop != aud:
        return False
    cond_icd = list(meta.get("icd10") or [])
    if cond_icd and icd_list and not _icd_lists_overlap(cond_icd, icd_list):
        return False
    return True


def _check_diagnosis_components(
    diagnosis_text: str,
    required: list[str],
) -> tuple[list[str], list[str]]:
    low = _norm(diagnosis_text)
    present: list[str] = []
    missing: list[str] = []
    markers = {
        "нозология": ("гэрб", "рефлюкс", "гастрит", "язв", "k21", "k29", "k25", "k26", "диспепс", "колит", "крон", "целиак", "k50", "k51", "k90", "k85", "k35", "панкреат", "аппендицит", "пневмон", "бронхит", "астм", "диабет", "гипертон", "инфаркт", "инсульт", "артрит", "анем"),
        "клиническая форма": ("форма", "неэрозив", "эрозив", "атроф", "поверхност", "катаральн", "флегмон"),
        "форма": ("форма", "лёгк", "легк", "умерен", "тяжел", "тяжёл", "катаральн", "флегмон", "гангрен"),
        "степень тяжести": ("степен", "лёгк", "легк", "средн", "тяжел", "тяжёл"),
        "фаза": ("фаз", "обострен", "ремисс", "хроническ", "остр"),
        "осложнения": ("осложнен", "кровотеч", "стеноз", "перфора", "без ослож", "перитонит", "абсцесс"),
        "источник": ("источник", "язв", "варикоз", "эроз", "диvert", "диверт"),
        "механизм": ("механизм", "удар", "паден", "дтп", "нож", "огнестр"),
        "локализация": (
            "локал", "антрал", "луковиц", "желудк", "двенадцат", "l1", "l2", "l3", "l4", "илеальн",
            "толстокиш", "долев", "сегмент", "бедрен", "подколен", "берцов", "голен", "нижн",
            "конечност", "права", "лева", "вены", "вен ",
        ),
        "стадия": ("стади", "tnm", "стад ", "iia", "iiia", "iv ", "реканализа", "острая", "подостра", "хроническ"),
        "h.pylori": ("нр", "hp", "helicobacter", "хеликобактер"),
        "этиологический фактор": ("этиолог", "нр", "нпвп", "стресс"),
        "активность": ("активност", "hp 3", "воспален", "обострен", "ремисс"),
        "атрофия": ("атроф", "olga"),
        "метаплазия": ("метаплаз", "olgim"),
        "острый или хронический характер": ("остр", "хроническ"),
        "протяженность": ("протяжен", "дистальн", "тотальн", "леоколит", "панколит", "проктит", "левая"),
        "протяженность поражения кишечника": ("проктит", "панколит", "левая", "тотальн", "левосторон"),
        "фазу течения - обострение или ремиссия": ("обострен", "ремисс", "фаз"),
        "характер течения": ("остр", "хроническ", "рецидив", "непрерывн"),
        "тяжесть текущего обострения в соответствии с пиаяк": ("тяжел", "тяжёл", "лёгк", "легк", "умерен", "пиаяк"),
        "тяжесть": ("тяжел", "тяжёл", "среднетяж", "лёгк", "легк", "умерен"),
        "диагноз в соответствии с парижской педиатрической классификацией бк": ("l1", "l2", "l3", "a1", "b1", "b2", "b3", "локализац", "илеальн", "толстокиш"),
        "вариант течения": ("вариант", "течени", "рекуррент"),
        "период": ("период", "ремисс", "обострен"),
        "гистологическая стадия": ("гистолог", "стади", "marsh"),
        "тип": ("тип", "1 тип", "2 тип", "инсулин", "сахарн"),
        "компенсация": ("компенсац", " hba1c", "гликир", "сахар"),
        "контроль": ("контрол", "обострен", "ремисс", "стабильн"),
        "риск": ("риск", "фактор", "осложнен"),
        "функциональный класс": ("функциональн", "класс", "фк ", "nyha"),
        "гистология": ("гистолог", "диффузн", "фолликуляр", "ходжкин"),
        "срок": ("срок", "недел", "триместр", "гестаци", "от 20", "от 19", "от 18"),
        "возрастная группа": ("возраст", "дет", "подрост", "новорожд"),
        "бактериовыделение": ("бактериовыдел", "микобакт", "бацилл"),
        "эпизод": ("эпизод", "депрессив", "маниакальн", "рекуррент"),
        "функция": ("функци", "гипотиреоз", "гипертиреоз", "ттг"),
        "степень": ("степен", "i ст", "ii ст", "iii ст", "лёгк", "легк", "умерен", "тяжел"),
        "частота": ("частот", "приступ", "эпизод", "раз в"),
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
            if not found and key == "срок" and re.search(r"\d{2}\.\d{2}\.\d{4}", low):
                found = True
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
        "rule_source": rule.get("rule_source"),
        "generated_from_summary": rule.get("generated_from_summary"),
        "condition_id": rule.get("condition_id"),
    }

    if rule_type == "population_mismatch":
        expected = (rule.get("expected_population") or "").lower()
        actual = (ctx.get("adult_or_child") or "").lower()
        if expected and actual and expected != actual:
            finding["passed"] = False
            finding["severity"] = "critical"
            finding["message_ru"] = (
                f"Протокол рассчитан на пациентов ({population_ru(expected)}), "
                f"в КЗ указана аудитория: {population_ru(actual)}."
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
                finding["message_ru"] = localize_message_ru(
                    "Не найдено достаточных диагностических критериев в тексте КЗ "
                    f"({rule.get('description_ru') or rule_title_ru(str(rule.get('rule_id') or ''), rule)})."
                )
        else:
            finding["passed"] = len(hits) == len(criteria) and bool(criteria)
            if not finding["passed"]:
                finding["message_ru"] = "Не все диагностические критерии подтверждены текстом КЗ."
        finding["criteria_matched"] = len(hits)
        return finding

    if rule_type == "required_exam":
        exam_raw = str(rule.get("exam") or "")
        exam = _norm(exam_raw)
        if exam and exam not in _norm(text):
            from .semantic_rule_fallback import semantic_presence_check

            sem = semantic_presence_check(text, exam_raw, rule=rule)
            if sem.get("matched"):
                finding["semantic_match"] = True
                finding["semantic_method"] = sem.get("method")
                finding["semantic_confidence"] = sem.get("confidence")
                finding["message_ru"] = (
                    f"Обследование найдено семантически ({sem.get('method')}): {rule.get('exam')}."
                )
            else:
                finding["passed"] = False
                finding["message_ru"] = (
                    f"В КЗ не упомянуто обязательное обследование: {rule.get('exam')}."
                )
        return finding

    if rule_type == "keyword_presence":
        kw_raw = str(rule.get("keyword") or "")
        kw = _norm(kw_raw)
        if kw and kw not in _norm(text):
            from .semantic_rule_fallback import semantic_presence_check

            sem = semantic_presence_check(text, kw_raw, rule=rule)
            if sem.get("matched"):
                finding["semantic_match"] = True
                finding["semantic_method"] = sem.get("method")
                finding["semantic_confidence"] = sem.get("confidence")
                finding["message_ru"] = (
                    rule.get("message_ru")
                    or f"Упоминание найдено семантически ({sem.get('method')}): {rule.get('keyword')}."
                )
            else:
                finding["passed"] = False
                finding["message_ru"] = (
                    rule.get("message_ru") or f"Ожидалось упоминание: {rule.get('keyword')}."
                )
        return finding

    if rule_type == "red_flag_rule":
        keywords = list(rule.get("keywords") or [])
        if rule.get("keyword"):
            keywords.append(str(rule.get("keyword")))
        low = _norm(text + " " + diagnosis)
        hit = any(_norm(k) in low for k in keywords if k)
        finding["passed"] = hit
        if not hit:
            finding["severity"] = rule.get("severity") or "high"
            finding["passed"] = False
            finding["message_ru"] = (
                rule.get("message_ru")
                or "Красный флаг протокола не отражён в КЗ: " + (keywords[0] if keywords else "")
            )
        return finding

    if rule_type in ("diagnosis_structure_rule",):
        required = list(rule.get("required_components") or rule.get("expected_items") or [])
        present, missing = _check_diagnosis_components(diagnosis, required)
        finding["present"] = present
        finding["missing"] = missing
        if missing:
            finding["passed"] = False
            finding["message_ru"] = "В формулировке диагноза не хватает компонентов: " + ", ".join(missing)
        return finding

    return finding


def _condition_meta(cid: str, conditions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cond = conditions.get(cid)
    if cond:
        return cond
    cdef = CONDITION_BY_ID.get(cid)
    if not cdef:
        return None
    return {
        "condition_id": cid,
        "condition": cid.replace("_", " "),
        "population": "any",
    }


def _conditions_from_matched(matched_protocols: list[dict[str, Any]] | None) -> list[str]:
    from .rules_from_path import infer_path_condition

    out: list[str] = []
    for sp in matched_source_paths(matched_protocols):
        hit = infer_path_condition(sp)
        if hit:
            out.append(hit[0])
    return list(dict.fromkeys(out))


def _augment_rules_map(
    rules_map: dict[str, list[dict[str, Any]]],
    matched_protocols: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Runtime path-правила для matched PDF без статического файла."""
    paths = matched_source_paths(matched_protocols)
    if not paths:
        return rules_map

    covered_paths: set[str] = set()
    for rules in rules_map.values():
        for rule in rules:
            src = rule.get("source") or {}
            sp = (src.get("source_path") or "").replace("\\", "/").strip()
            if sp:
                covered_paths.add(sp)

    from .rules_from_path import extract_path_rules

    augmented = {cid: list(rules) for cid, rules in rules_map.items()}
    for sp in paths:
        if sp in covered_paths:
            continue
        pdf_hash = sha256(sp.encode()).hexdigest()[:8]
        protocol_id = f"proto_{pdf_hash}"
        for cid, rules in extract_path_rules(
            sp, protocol_id=protocol_id, rule_id_prefix=pdf_hash
        ).items():
            bucket = augmented.setdefault(cid, [])
            seen = {r.get("rule_id") for r in bucket}
            for rule in rules:
                rid = rule.get("rule_id")
                if rid and rid in seen:
                    continue
                bucket.append(rule)
                if rid:
                    seen.add(rid)
    return augmented


def _resolve_target_conditions(
    consult_facts: dict[str, Any],
    *,
    condition_ids: list[str] | None,
    matched_protocols: list[dict[str, Any]] | None,
    conditions: dict[str, dict[str, Any]],
    rules_map: dict[str, list[dict[str, Any]]],
) -> list[str]:
    cons = consult_facts.get("consultation") or {}
    hints = list(cons.get("conditions_hint") or [])
    icd_list = [str(x).upper() for x in (cons.get("icd10") or [])]

    if condition_ids:
        base = list(condition_ids)
    elif hints:
        base = hints
    else:
        base = infer_conditions_hints(_text_blob(consult_facts), icd_list)

    for cid in _conditions_from_matched(matched_protocols):
        if cid not in base:
            base.append(cid)

    if not base and matched_protocols:
        base = [cid for cid in rules_map if rules_map.get(cid)]

    if not base:
        base = list(conditions.keys()) or list(rules_map.keys())

    if extra_summary := [cid for cid, rs in rules_map.items() if any(r.get("rule_source") == "summary" for r in rs)]:
        for cid in extra_summary:
            if cid not in base:
                base.append(cid)

    filtered = [
        cid
        for cid in dict.fromkeys(base)
        if _condition_applies_to_consult(cid, conditions, consult_facts)
    ]
    return filtered


def collect_catalog_rules(
    consult_facts: dict[str, Any],
    *,
    condition_ids: list[str] | None = None,
    matched_protocols: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Список legacy-правил каталога для matched conditions (без прогона)."""
    conditions = load_conditions()
    rules_map = _augment_rules_map(load_rules_by_condition(), matched_protocols)
    target = _resolve_target_conditions(
        consult_facts,
        condition_ids=condition_ids,
        matched_protocols=matched_protocols,
        conditions=conditions,
        rules_map=rules_map,
    )
    out: list[dict[str, Any]] = []
    for cid in target:
        for rule in filter_rules_for_matched_protocols(rules_map.get(cid) or [], matched_protocols):
            r = dict(rule)
            r.setdefault("rule_source", "legacy")
            out.append(r)
    return out


def run_rule_checker(
    consult_facts: dict[str, Any],
    *,
    condition_ids: list[str] | None = None,
    matched_protocols: list[dict[str, Any]] | None = None,
    extra_rules: list[dict[str, Any]] | None = None,
    include_catalog: bool = True,
    skip_rule_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Проверка КЗ по правилам каталога; возвращает findings и сводку."""
    conditions = load_conditions()
    rules_map = (
        _augment_rules_map(load_rules_by_condition(), matched_protocols)
        if include_catalog
        else {}
    )
    skip_rule_ids = frozenset(skip_rule_ids or ())
    if extra_rules:
        for er in extra_rules:
            cid = str(er.get("condition_id") or er.get("summary_condition_id") or "_summary")
            rules_map.setdefault(cid, []).append(er)
    target = _resolve_target_conditions(
        consult_facts,
        condition_ids=condition_ids,
        matched_protocols=matched_protocols,
        conditions=conditions,
        rules_map=rules_map,
    )

    all_findings: list[dict[str, Any]] = []
    checked_conditions: list[str] = []

    for cid in target:
        cond = _condition_meta(cid, conditions)
        rules = rules_map.get(cid) or []
        if not cond and not rules:
            continue
        checked_conditions.append(cid)
        ctx = consult_facts.get("patient_context") or {}
        if cond:
            pop = cond.get("population")
            if pop and pop != "any" and ctx.get("adult_or_child") and pop != ctx.get("adult_or_child"):
                all_findings.append(
                    {
                        "rule_id": f"{cid}_population_guard",
                        "rule_type": "population_mismatch",
                        "severity": "info",
                        "passed": True,
                        "skipped": True,
                        "not_applicable": True,
                        "message_ru": localize_message_ru(
                            f"Нозология «{cond.get('condition')}» - протокол для "
                            f"{population_ru(str(pop))}, в КЗ аудитория "
                            f"{population_ru(str(ctx.get('adult_or_child') or ''))}; проверка пропущена."
                        ),
                        "source": cond.get("protocol_reference"),
                    }
                )
                continue
        for rule in filter_rules_for_matched_protocols(rules, matched_protocols):
            if str(rule.get("rule_id") or "") in skip_rule_ids:
                continue
            proto = legacy_rule_to_protocol_rule(rule)
            if not rule_applicable_to_patient(proto, ctx):
                all_findings.append(
                    {
                        "rule_id": rule.get("rule_id"),
                        "rule_type": rule.get("rule_type"),
                        "severity": "info",
                        "passed": True,
                        "skipped": True,
                        "not_applicable": True,
                        "message_ru": "Правило неприменимо по возрасту/полу/беременности - не учитывается в score.",
                        "source": rule.get("source"),
                    }
                )
                continue
            all_findings.append(_run_rule(rule, consult_facts))

    failed = [f for f in all_findings if not f.get("passed") and not f.get("skipped")]
    critical = [f for f in failed if f.get("severity") == "critical"]

    compliance_pct = None
    if all_findings:
        scored = [f for f in all_findings if not f.get("skipped")]
        passed_n = sum(1 for f in scored if f.get("passed"))
        compliance_pct = round(100.0 * passed_n / len(scored), 1) if scored else None

    for f in all_findings:
        if not f.get("title_ru"):
            f["title_ru"] = rule_title_ru(str(f.get("rule_id") or ""), f)
        if f.get("message_ru"):
            f["message_ru"] = localize_message_ru(str(f["message_ru"]))

    return {
        "checked_conditions": checked_conditions,
        "findings": all_findings,
        "failed_count": len(failed),
        "critical_count": len(critical),
        "missing_required_items": [f.get("message_ru") for f in failed if f.get("message_ru")],
        "rules_compliance_pct": compliance_pct,
        "method": "deterministic_rules_v3_all_catalog",
        "matched_source_paths": sorted(matched_source_paths(matched_protocols)),
        "rules_filtered_by_protocol": bool(matched_protocols),
    }
