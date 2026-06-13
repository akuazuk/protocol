"""Проверка красных флагов безопасности в КЗ (ТЗ §17 / improve_kz §14)."""
from __future__ import annotations

from typing import Any

from .consult_config import load_red_flags
from .consult_schema import ConsultationDocument, SafetyAssessment, SafetyCapInfo

_ISSUE_TYPE_MAP = {
    "possible_malignancy": "possible_malignancy",
    "thrombosis": "thrombosis",
    "severe_infection": "severe_infection",
    "systemic_autoimmune": "red_flag",
    "gi_bleeding_anemia": "red_flag",
    "drug_safety": "drug_safety",
}

_CATEGORY_RU = {
    "possible_malignancy": "подозрение на онкологию",
    "thrombosis": "тромбоз/тромбофлебит",
    "systemic_autoimmune": "системное аутоиммунное",
    "severe_infection": "тяжёлая инфекция",
    "gi_bleeding_anemia": "ЖКТ-кровотечение/анемия",
    "drug_safety": "безопасность лекарственной терапии",
}

_HANDLING_MARKERS = (
    "консультац", "направлен", "госпитализац", "маршрут", "дообследован",
    "обследован", "повторн", "контрол", "узи", "биопси", "онколог",
    "антикоагулянт", "антибактериальн", "ревматолог", "фотозащит",
    "колоноскоп", "анализ", "экг", "гастро",
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
        s.non_drug_recommendations or "",
    ]
    parts += [d.raw_text for d in doc.follow_up if d.raw_text]
    parts += [m.raw_text for m in doc.medications if m.raw_text]
    return "\n".join(parts).lower()


def _match_required_actions(actions_blob: str, required: list[str]) -> tuple[int, int]:
    if not required:
        return 0, 0
    hits = 0
    for act in required:
        low = str(act).lower()
        if low in actions_blob:
            hits += 1
            continue
        if any(tok in actions_blob for tok in low.split() if len(tok) > 4):
            hits += 1
            continue
        if any(mk in actions_blob for mk in _HANDLING_MARKERS if mk in low):
            hits += 1
            continue
        # русские падежные формы: общий корень ≥6 символов
        stem = low[: max(6, len(low) - 2)]
        if len(stem) >= 5 and stem in actions_blob:
            hits += 1
    return hits, len(required)


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
        required = [str(a) for a in (cfg.get("required_actions") or cfg.get("expected_actions") or [])]
        hits, total = _match_required_actions(actions, required)
        if hits >= total and total > 0:
            status = "handled"
        elif severity == "critical" and any(mk in actions for mk in _HANDLING_MARKERS):
            status = "handled"
        elif hits >= max(1, (total + 1) // 2):
            status = "partially_handled"
        elif any(mk in actions for mk in _HANDLING_MARKERS):
            status = "partially_handled"
        else:
            status = "not_handled"
        out.append(
            SafetyAssessment(
                issue_type=_ISSUE_TYPE_MAP.get(flag_id, "red_flag"),  # type: ignore[arg-type]
                severity=severity if severity in ("low", "medium", "high", "critical") else "medium",  # type: ignore[arg-type]
                finding_text=f"Найден признак: «{hit}» (категория: {_CATEGORY_RU.get(flag_id, flag_id)}).",
                expected_action=", ".join(required) or None,
                actual_action=(
                    f"Выполнено {hits}/{total} ожидаемых действий."
                    if hits else None
                ),
                status=status,  # type: ignore[arg-type]
            )
        )
    return out


def has_unhandled_critical(safety: list[SafetyAssessment]) -> bool:
    return any(
        s.severity == "critical" and s.status not in ("handled", "partially_handled")
        for s in safety
    )


def apply_safety_cap_to_score(
    overall: float | None,
    safety: list[SafetyAssessment],
) -> tuple[float | None, SafetyCapInfo]:
    """Применяет cap_if_unhandled из red_flags.yaml."""
    if overall is None:
        return None, SafetyCapInfo(applied=False)
    red_flags = load_red_flags()
    cap = float(overall)
    applied = False
    reason: str | None = None
    cap_val: float | None = None
    for s in safety:
        if s.status in ("handled", "partially_handled"):
            continue
        for flag_id, cfg in red_flags.items():
            if s.issue_type != _ISSUE_TYPE_MAP.get(flag_id, "red_flag"):
                continue
            if flag_id == "possible_malignancy" and s.issue_type != "possible_malignancy":
                continue
            if flag_id == "thrombosis" and s.issue_type != "thrombosis":
                continue
            if flag_id == "drug_safety" and s.issue_type != "drug_safety":
                continue
            limit = cfg.get("cap_if_unhandled")
            if isinstance(limit, (int, float)):
                if cap > float(limit):
                    cap = float(limit)
                    applied = True
                    cap_val = float(limit)
                    reason = f"Необработанный флаг «{flag_id}» - safety cap {limit}%"
    return cap, SafetyCapInfo(applied=applied, reason=reason, cap_value=cap_val)
