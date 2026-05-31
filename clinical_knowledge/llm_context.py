"""Форматирование результатов rule checker для промпта consult-review."""
from __future__ import annotations

from typing import Any


def format_clinical_rules_for_llm(clinical_rules: dict[str, Any] | None) -> str:
    """Текстовый блок для extra_context в _consult_review_synthesize."""
    if not clinical_rules or not isinstance(clinical_rules, dict):
        return ""

    rc = clinical_rules.get("rules_check") or {}
    if not isinstance(rc, dict):
        return ""

    lines: list[str] = [
        "ДЕТЕРМИНИРОВАННАЯ ПРОВЕРКА ПО ПРАВИЛАМ ПРОТОКОЛА (MVP, до оценки модели):",
        "Учитывай эти выводы при формулировке criteria и limitations_ru; не противоречь им без явного обоснования в limitations_ru.",
    ]

    pct = rc.get("rules_compliance_pct")
    if isinstance(pct, (int, float)):
        lines.append(f"Сводный балл по правилам: {pct}%.")

    matched = clinical_rules.get("matched_protocols") or []
    if matched:
        titles = [
            (m.get("title") or m.get("protocol_id") or "")[:100]
            for m in matched[:4]
            if isinstance(m, dict)
        ]
        if titles:
            lines.append("Подобранные карточки протоколов: " + "; ".join(titles) + ".")

    missing = [m for m in (rc.get("missing_required_items") or []) if m]
    if missing:
        lines.append("Обязательные пробелы / замечания:")
        for m in missing[:10]:
            lines.append(f"- {m}")

    failed = [
        f
        for f in (rc.get("findings") or [])
        if isinstance(f, dict) and not f.get("passed") and f.get("severity") == "critical"
    ]
    if failed:
        lines.append("Критические несоответствия:")
        for f in failed[:6]:
            msg = f.get("message_ru") or f.get("rule_id") or ""
            if msg:
                lines.append(f"- {msg}")

    facts = clinical_rules.get("consult_facts") or {}
    cons = facts.get("consultation") or {}
    icd = cons.get("icd10") or []
    if icd:
        lines.append("МКБ из КЗ (эвристика): " + ", ".join(str(c) for c in icd[:8]))

    if len(lines) <= 2:
        return ""
    return "\n".join(lines) + "\n"
