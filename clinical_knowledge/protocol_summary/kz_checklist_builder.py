"""Детерминированная сборка kz_checklist из структурированных полей протокола.

Ключевая идея Э1 методологии оценки КЗ 2.0 (docs/plans/2026-07-22-kz-scoring-methodology-v1.md):
kz_checklist (must/should/conditional/warnings) - это диагноз-специфичный «эталон КЗ»,
из которого движок оценки берёт «что обязано быть в заключении по этому диагнозу».

Мы НЕ используем LLM и НЕ выдумываем пункты: каждый пункт выводится из уже извлечённых
и привязанных к цитате протокола полей (`required_exams`, `diagnostic_criteria`,
`treatment`, `follow_up`, `red_flags`, `routing`, `diagnosis_structure`). Поэтому результат
машинно-сгенерирован и source-anchored по построению (guardrail вместо ручного ревью).

Работает на dict (JSON карточки), чтобы не тянуть pydantic там, где он недоступен.
"""
from __future__ import annotations

import re
from typing import Any

_MAX_ITEM_LEN = 160
_MAX_PER_BUCKET = 14


def _clean(text: Any) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:_MAX_ITEM_LEN].rstrip(" .;,")


def _short(text: Any, limit: int = 90) -> str:
    s = _clean(text)
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _dedup_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        key = it.lower()
        if not it or key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= _MAX_PER_BUCKET:
            break
    return out


def _required_if_text(item: dict[str, Any]) -> str:
    ri = item.get("required_if") or []
    if isinstance(ri, list) and ri:
        return ", ".join(_short(x, 60) for x in ri[:3])
    return ""


def _applicability_text(item: dict[str, Any]) -> str:
    appl = item.get("applicability") or {}
    if not isinstance(appl, dict):
        return ""
    bits: list[str] = []
    pop = appl.get("population") or []
    if isinstance(pop, list) and pop and "unknown" not in pop:
        bits.append("/".join(str(p) for p in pop))
    if appl.get("sex") and appl.get("sex") not in ("any", "unknown"):
        bits.append(str(appl.get("sex")))
    if appl.get("pregnancy") == "required":
        bits.append("беременность")
    return ", ".join(bits)


def build_kz_checklist(cond: dict[str, Any]) -> dict[str, list[str]]:
    """Собрать kz_checklist (must/should/conditional/warnings) из полей condition-dict."""
    must: list[str] = []
    should: list[str] = []
    conditional: list[str] = []
    warnings: list[str] = []

    # --- Диагноз и его структура (ось A: документация) ---
    if cond.get("icd10_codes"):
        must.append("Диагноз сформулирован и указан код МКБ-10")
    ds = cond.get("diagnosis_structure") or {}
    if isinstance(ds, dict):
        for comp in ds.get("required_components") or []:
            if isinstance(comp, dict) and comp.get("name"):
                must.append(f"В формулировке диагноза отражено: {_short(comp['name'])}")
        for comp in ds.get("optional_components") or []:
            if isinstance(comp, dict) and comp.get("name"):
                should.append(f"Желательно в диагнозе: {_short(comp['name'])}")

    # --- Диагностические критерии (ось B: обоснование Dx) ---
    dc = cond.get("diagnostic_criteria") or {}
    if isinstance(dc, dict):
        for cr in dc.get("required") or []:
            if isinstance(cr, dict) and cr.get("text"):
                must.append(f"Обоснование диагноза: {_short(cr['text'])}")
        for cr in dc.get("optional") or []:
            if isinstance(cr, dict) and cr.get("text"):
                should.append(f"Учтён критерий: {_short(cr['text'])}")

    # --- Обследования (ось B) ---
    def _exam_line(ex: dict[str, Any]) -> str:
        return _short(ex.get("name"))

    for ex in cond.get("required_exams") or []:
        if not isinstance(ex, dict) or not ex.get("name"):
            continue
        level = str(ex.get("requirement_level") or "required")
        rif = _required_if_text(ex)
        if rif or level == "conditional":
            suffix = f" (если {rif})" if rif else " (по показаниям)"
            conditional.append(f"Обследование: {_exam_line(ex)}{suffix}")
        elif level == "required":
            must.append(f"Назначено/выполнено обследование: {_exam_line(ex)}")
        else:  # recommended / optional
            should.append(f"Рекомендуемое обследование: {_exam_line(ex)}")
    for ex in cond.get("conditional_exams") or []:
        if isinstance(ex, dict) and ex.get("name"):
            rif = _required_if_text(ex)
            suffix = f" (если {rif})" if rif else " (по показаниям)"
            conditional.append(f"Обследование: {_exam_line(ex)}{suffix}")

    # --- Лечение (ось B) ---
    tr = cond.get("treatment") or {}
    if isinstance(tr, dict):
        has_drugs = bool(tr.get("drugs") or tr.get("drug_groups"))
        if has_drugs:
            must.append("Назначена медикаментозная терапия согласно протоколу")
        for g in tr.get("drug_groups") or []:
            if isinstance(g, dict) and g.get("drug_group"):
                should.append(f"Группа препаратов: {_short(g['drug_group'])}")
        for d in tr.get("drugs") or []:
            if not isinstance(d, dict):
                continue
            name = d.get("drug_name") or d.get("active_substance") or d.get("drug_group")
            if not name:
                continue
            appl = _applicability_text(d)
            if appl:
                conditional.append(f"Препарат при [{appl}]: {_short(name)}")
            else:
                should.append(f"Препарат: {_short(name)}")
        for nd in tr.get("non_drug") or []:
            if isinstance(nd, dict) and nd.get("text"):
                should.append(f"Немедикаментозно: {_short(nd['text'])}")
        for pr in tr.get("procedures") or []:
            if isinstance(pr, dict) and pr.get("name"):
                should.append(f"Процедура: {_short(pr['name'])}")
        for sg in tr.get("surgery") or []:
            if isinstance(sg, dict) and sg.get("name"):
                conditional.append(f"Хирургия (по показаниям): {_short(sg['name'])}")

    # --- Наблюдение и маршрутизация ---
    for fu in cond.get("follow_up") or []:
        if not isinstance(fu, dict) or not fu.get("text"):
            continue
        rif = _required_if_text(fu)
        timing = _short(fu.get("timing"), 40) if fu.get("timing") else ""
        base = _short(fu["text"])
        tail = f" (срок: {timing})" if timing else ""
        if rif:
            conditional.append(f"Наблюдение: {base}{tail} (если {rif})")
        else:
            should.append(f"План наблюдения: {base}{tail}")
    for key in ("routing", "hospitalization"):
        for rt in cond.get(key) or []:
            if not isinstance(rt, dict) or not rt.get("text"):
                continue
            rif = _required_if_text(rt)
            suffix = f" (если {rif})" if rif else " (по показаниям)"
            conditional.append(f"Маршрутизация: {_short(rt['text'])}{suffix}")

    # --- Красные флаги (ось C: безопасность) ---
    for rf in cond.get("red_flags") or []:
        if not isinstance(rf, dict) or not rf.get("text"):
            continue
        sev = str(rf.get("severity") or "medium")
        actions = rf.get("expected_actions") or []
        act_txt = ", ".join(_short(a, 50) for a in actions[:3]) if actions else ""
        prefix = "КРИТИЧНО: " if sev == "critical" else ("Важно: " if sev == "high" else "")
        line = f"{prefix}при «{_short(rf['text'], 80)}»"
        if act_txt:
            line += f" - {act_txt}"
        warnings.append(_clean(line))

    return {
        "must_have": _dedup_keep_order(must),
        "should_have": _dedup_keep_order(should),
        "conditional": _dedup_keep_order(conditional),
        "warnings": _dedup_keep_order(warnings),
    }


def checklist_is_nonempty(checklist: dict[str, list[str]] | None) -> bool:
    if not checklist or not isinstance(checklist, dict):
        return False
    return any(checklist.get(k) for k in ("must_have", "should_have", "conditional", "warnings"))


def condition_has_source_fields(cond: dict[str, Any]) -> bool:
    """Есть ли из чего собирать checklist (иначе оставляем пустым)."""
    if cond.get("required_exams") or cond.get("conditional_exams"):
        return True
    if cond.get("red_flags") or cond.get("follow_up") or cond.get("routing") or cond.get("hospitalization"):
        return True
    tr = cond.get("treatment") or {}
    if isinstance(tr, dict) and any(
        tr.get(k) for k in ("drugs", "drug_groups", "non_drug", "procedures", "surgery")
    ):
        return True
    dc = cond.get("diagnostic_criteria") or {}
    if isinstance(dc, dict) and (dc.get("required") or dc.get("optional")):
        return True
    ds = cond.get("diagnosis_structure") or {}
    if isinstance(ds, dict) and ds.get("required_components"):
        return True
    return False
