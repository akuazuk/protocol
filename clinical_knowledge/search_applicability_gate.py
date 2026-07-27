"""Applicability-gate поиска протоколов (ТЗ №2, Workstream A - correctness до restyle).

Главный инвариант (§3 P0):
    Нельзя показывать результат как «рекомендуемый», если возрастная группа, особое
    состояние, условия помощи, применимость МКБ или актуальность документа не
    подтверждены. Детский протокол не может стать рекомендуемым Top-1 для взрослого
    (или аудиторно-неопределённого) запроса без явного подтверждения детской аудитории.

Модуль - чистый re-rank + классификация статуса поверх готового списка карточек
(match_protocol_cards). Не ослабляет safety: только понижает неподтверждённое и
проставляет честный статус/объяснение. Включается флагом ``SEARCH_APPLICABILITY_GATE``
(по умолчанию ON), безопасен и аддитивен.
"""
from __future__ import annotations

import os
from typing import Any

from .applicability import assess_card_applicability, infer_card_population

# Статусы результата (§A2)
STATUS_EXACT = "exact_match"
STATUS_ICD = "icd_match"
STATUS_POSSIBLE = "possible"
STATUS_CLARIFY = "needs_clarification"
STATUS_NOT_FOR_AUDIENCE = "not_for_audience"
STATUS_OUTDATED = "outdated"

STATUS_LABELS_RU = {
    STATUS_EXACT: "Точное совпадение",
    STATUS_ICD: "Подходит по коду МКБ",
    STATUS_POSSIBLE: "Возможное совпадение",
    STATUS_CLARIFY: "Требуется уточнение",
    STATUS_NOT_FOR_AUDIENCE: "Не подходит по аудитории",
    STATUS_OUTDATED: "Устаревший документ",
}

# «Рекомендуем» разрешено только выше этого порога уверенности И при подтверждённой
# применимости (§A2).
RECOMMEND_MIN_SCORE = 60.0


def gate_enabled() -> bool:
    return os.environ.get("SEARCH_APPLICABILITY_GATE", "1").strip().lower() in ("1", "true", "yes", "on")


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _query_icd_roots(icd_query_codes: list[str] | None) -> set[str]:
    return {_icd_root(c) for c in (icd_query_codes or []) if c}


def _card_icd_roots(card: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    primary = card.get("icd10_primary")
    if primary:
        roots.add(_icd_root(str(primary)))
    for f in card.get("icd_fit") or []:
        code = f.get("code") if isinstance(f, dict) else None
        if code:
            roots.add(_icd_root(str(code)))
    for code in card.get("matched_icd_codes") or []:
        if code:
            roots.add(_icd_root(str(code)))
    return roots


def _is_outdated(card: dict[str, Any]) -> bool:
    status = str(card.get("status") or card.get("doc_status") or "").lower()
    if status in ("obsolete", "deprecated", "expired", "withdrawn", "устаревший"):
        return True
    appr = card.get("approval")
    if isinstance(appr, dict):
        vt = str(appr.get("valid_to") or "").strip()
        # формат YYYY или YYYY-MM-DD; сравниваем по году
        if vt[:4].isdigit() and int(vt[:4]) < 2000:
            return True
    return False


def classify_result(
    card: dict[str, Any],
    patient: dict[str, Any] | None,
    icd_query_codes: list[str] | None,
    *,
    pediatric_signal: bool = False,
) -> dict[str, Any]:
    """Классифицировать один результат: статус, recommended, объяснения (§A2, §A3)."""
    patient = patient or {}
    audience = (patient.get("adult_or_child") or "").lower() or None
    card_pop = infer_card_population(card)
    population_specific = card_pop in ("child", "children", "pediatric", "adult")
    is_child_card = card_pop in ("child", "children", "pediatric")

    appl, match_reasons, mismatch_reasons = assess_card_applicability(card, patient)

    q_roots = _query_icd_roots(icd_query_codes)
    c_roots = _card_icd_roots(card)
    icd_exact = bool(q_roots and c_roots and (q_roots & c_roots))
    icd_query_present = bool(q_roots)

    raw_score = card.get("match_score")
    if raw_score is None:
        raw_score = card.get("score")
    if raw_score is None:
        raw_score = card.get("confidence_score")
        if raw_score is not None and float(raw_score) <= 1.0:
            raw_score = float(raw_score) * 100.0
    score = float(raw_score or 0.0)

    reasons: list[str] = []
    if icd_exact:
        reasons.append(f"Совпал код МКБ: {', '.join(sorted(q_roots & c_roots))}.")
    reasons.append(f"Аудитория документа: {_pop_label(card_pop)}.")
    setting = str(card.get("care_setting") or card.get("setting") or "").strip()
    if setting:
        reasons.append(f"Условия помощи: {setting}.")
    year = _doc_year(card)
    if year:
        reasons.append(f"Год документа: {year}.")
    reasons.extend(match_reasons)
    reasons.extend(mismatch_reasons)

    # --- определить статус (порядок важен) ---
    if _is_outdated(card):
        status = STATUS_OUTDATED
    elif appl == "not_applicable":
        status = STATUS_NOT_FOR_AUDIENCE
    elif population_specific and audience is None and not (is_child_card and pediatric_signal):
        # аудитория не подтверждена для population-specific документа -> уточнение
        status = STATUS_CLARIFY
        reasons.append(
            "Аудитория запроса не подтверждена, а документ рассчитан на конкретную "
            "возрастную группу - требуется уточнение.",
        )
    elif icd_exact and appl in ("applicable", "unknown"):
        status = STATUS_EXACT
    elif icd_exact or (icd_query_present and c_roots):
        status = STATUS_ICD
    else:
        status = STATUS_POSSIBLE

    recommended = (
        status in (STATUS_EXACT, STATUS_ICD)
        and appl == "applicable"
        and score >= RECOMMEND_MIN_SCORE
    )

    return {
        "status": status,
        "status_label_ru": STATUS_LABELS_RU[status],
        "recommended": recommended,
        "applicability": appl,
        "population": card_pop,
        "why_reasons": [r for r in reasons if r],
        "requires_clarification": status == STATUS_CLARIFY,
    }


def _pop_label(pop: str) -> str:
    return {
        "child": "детское население",
        "children": "детское население",
        "pediatric": "детское население",
        "adult": "взрослое население",
    }.get(pop, "без возрастных ограничений")


def _doc_year(card: dict[str, Any]) -> str | None:
    appr = card.get("approval")
    if isinstance(appr, dict):
        for key in ("document_year", "valid_from", "approval_date"):
            v = str(appr.get(key) or "").strip()
            if v[:4].isdigit():
                return v[:4]
    y = str(card.get("document_year") or "").strip()
    return y[:4] if y[:4].isdigit() else None


# Ранг для сортировки (меньше = выше). Гарантирует, что неподтверждённое
# population-specific (особенно детское) не станет Top-1 над нейтральным/подтверждённым.
_STATUS_RANK = {
    STATUS_EXACT: 0,
    STATUS_ICD: 1,
    STATUS_POSSIBLE: 2,
    STATUS_CLARIFY: 3,
    STATUS_NOT_FOR_AUDIENCE: 4,
    STATUS_OUTDATED: 5,
}


def _sort_key(item: dict[str, Any]) -> tuple:
    g = item.get("_gate") or {}
    status = g.get("status", STATUS_POSSIBLE)
    rank = _STATUS_RANK.get(status, 2)
    is_child = g.get("population") in ("child", "children", "pediatric")
    # внутри needs_clarification детское население - ниже взрослого/нейтрального
    child_penalty = 1 if (status == STATUS_CLARIFY and is_child) else 0
    route_delta = float(g.get("clinical_route_delta") or 0.0)
    score = item.get("match_score")
    if score is None:
        score = item.get("score")
    if score is None:
        score = item.get("confidence_score")
        if score is not None and float(score) <= 1.0:
            score = float(score) * 100.0
    score = float(score or 0.0)
    return (rank, child_penalty, -route_delta, -score)


def apply_applicability_gate(
    cards: list[dict[str, Any]],
    patient: dict[str, Any] | None,
    icd_query_codes: list[str] | None,
    *,
    pediatric_signal: bool = False,
    keep_not_applicable: bool = False,
    query: str = "",
) -> list[dict[str, Any]]:
    """Классифицировать и пере-ранжировать результаты с applicability-gate.

    Аддитивно добавляет к каждой карточке поле ``_gate`` (status/recommended/why) и
    поднимает подтверждённые/нейтральные результаты выше неподтверждённых population-
    specific. Не удаляет карточки (кроме not_applicable при keep_not_applicable=False).
    """
    route_ids: list[str] = []
    if query:
        try:
            from .search_clinical_routing import detect_clinical_route_ids

            route_ids = detect_clinical_route_ids(query, list(icd_query_codes or []))
        except Exception:
            route_ids = []

    out: list[dict[str, Any]] = []
    for card in cards:
        g = classify_result(card, patient, icd_query_codes, pediatric_signal=pediatric_signal)
        if route_ids:
            try:
                from .search_clinical_routing import score_path_for_clinical_routes

                route_delta, route_matches = score_path_for_clinical_routes(
                    str(card.get("source_path") or card.get("path") or ""),
                    str(card.get("title") or ""),
                    route_ids=route_ids,
                )
                g["clinical_route_delta"] = route_delta
                g["clinical_route_matches"] = route_matches
                if route_delta > 0:
                    g["why_reasons"].append("Название соответствует клиническому запросу.")
                elif route_delta < 0:
                    g["why_reasons"].append("Название указывает на другой клинический контекст.")
                # ICD compatibility can come from a comorbidity mention. For a
                # recognised clinical route, the stronger "recommended" claim
                # additionally requires positive title/path evidence.
                if route_delta <= 0:
                    g["recommended"] = False
            except Exception:
                pass
        if g["status"] == STATUS_NOT_FOR_AUDIENCE and not keep_not_applicable:
            continue
        enriched = dict(card)
        enriched["_gate"] = g
        # аддитивные плоские поля для фронта
        enriched["result_status"] = g["status"]
        enriched["result_status_label"] = g["status_label_ru"]
        enriched["recommended"] = g["recommended"]
        enriched["why_reasons"] = g["why_reasons"]
        out.append(enriched)
    out.sort(key=_sort_key)
    return out
