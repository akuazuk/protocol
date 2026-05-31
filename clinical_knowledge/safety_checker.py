"""Проверка красных флагов безопасности в КЗ (ТЗ раздел 17).

Если найден критический красный флаг без маршрутизации/дообследования — итоговый
статус не должен быть выше manual_review_required (обрабатывается в scoring).
"""
from __future__ import annotations

from typing import Any

from .consult_config import load_red_flags
from .consult_schema import ConsultationDocument, SafetyAssessment

_ISSUE_TYPE_MAP = {
    "possible_malignancy": "possible_malignancy",
    "thrombosis": "thrombosis",
    "severe_infection": "severe_infection",
    "systemic_autoimmune": "red_flag",
    "gi_bleeding_anemia": "red_flag",
}

# Человекочитаемые названия категорий красных флагов (для отчёта/UI).
_CATEGORY_RU = {
    "possible_malignancy": "подозрение на онкологию",
    "thrombosis": "тромбоз/тромбофлебит",
    "systemic_autoimmune": "системное аутоиммунное",
    "severe_infection": "тяжёлая инфекция",
    "gi_bleeding_anemia": "ЖКТ-кровотечение/анемия",
}

# Маркеры «обработки» флага (маршрутизация / дообследование / повторная явка / контроль)
_HANDLING_MARKERS = (
    "консультац", "направлен", "госпитализац", "маршрут", "дообследован",
    "обследован", "повторн", "контрол", "узи", "биопси", "онколог",
    "антикоагулянт", "антибактериальн", "ревматолог",
)


def _consult_blob(doc: ConsultationDocument) -> str:
    s = doc.sections
    parts = [
        doc.raw_text or "",
        s.recommendations_exams or "", s.recommendations_treatment or "",
        s.general_recommendations or "", s.follow_up_text or "",
    ]
    return "\n".join(parts).lower()


def _action_blob(doc: ConsultationDocument) -> str:
    s = doc.sections
    parts = [
        s.recommendations_exams or "", s.recommendations_treatment or "",
        s.general_recommendations or "", s.follow_up_text or "",
    ]
    parts += [d.raw_text for d in doc.follow_up if d.raw_text]
    return "\n".join(parts).lower()


def run_safety_checks(doc: ConsultationDocument) -> list[SafetyAssessment]:
    """Возвращает список SafetyAssessment по найденным красным флагам."""
    red_flags: dict[str, dict[str, Any]] = load_red_flags()
    blob = _consult_blob(doc)
    actions = _action_blob(doc)
    out: list[SafetyAssessment] = []

    for flag_id, cfg in red_flags.items():
        keywords = [str(k).lower() for k in (cfg.get("keywords") or [])]
        hit = next((k for k in keywords if k and k in blob), None)
        if not hit:
            continue
        severity = str(cfg.get("severity") or "medium")
        expected = ", ".join(str(a) for a in (cfg.get("expected_actions") or []))
        handled = any(mk in actions for mk in _HANDLING_MARKERS)
        status = "handled" if handled else "not_handled"
        out.append(
            SafetyAssessment(
                issue_type=_ISSUE_TYPE_MAP.get(flag_id, "red_flag"),  # type: ignore[arg-type]
                severity=severity if severity in ("low", "medium", "high", "critical") else "medium",  # type: ignore[arg-type]
                finding_text=f"Найден признак: «{hit}» (категория: {_CATEGORY_RU.get(flag_id, flag_id)}).",
                expected_action=expected or None,
                actual_action=None if not handled else "В рекомендациях есть маршрутизация/дообследование/контроль.",
                status=status,  # type: ignore[arg-type]
            )
        )
    return out


def has_unhandled_critical(safety: list[SafetyAssessment]) -> bool:
    return any(s.severity == "critical" and s.status != "handled" for s in safety)
