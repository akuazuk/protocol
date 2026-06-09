"""Оценка готовности КЗ/Bundle к импорту в ЦИСЗ (программа испытаний МИС v.1.3-4).

Дополняет клиническую оценку по КП Минздрава; не заменяет валидацию ЦИСЗ.
"""
from __future__ import annotations

from typing import Any, Literal

from .fhir_bundle_inspect import detect_bundle_scenario, inspect_bundle_checks, inspect_text_checks
from .fhir_mis_test_matrix import MisCheckDef, checks_for_scenario

VerdictRu = Literal["ok", "warn", "fail", "na"]


def _verdict_ru(passed: bool, critical: bool) -> str:
    if passed:
        return "Хорошо"
    return "Критично" if critical else "Доработать"


def _verdict_class(passed: bool, critical: bool) -> VerdictRu:
    if passed:
        return "ok"
    return "fail" if critical else "warn"


def _score_from_checks(
    check_defs: tuple[MisCheckDef, ...],
    results: dict[str, bool],
) -> tuple[float, list[dict[str, Any]], int]:
    total_w = sum(c.weight for c in check_defs)
    if total_w <= 0:
        return 0.0, [], 0
    earned = 0.0
    items: list[dict[str, Any]] = []
    critical_fail = 0
    for cdef in check_defs:
        passed = bool(results.get(cdef.check_id))
        if passed:
            earned += cdef.weight
        elif cdef.critical:
            critical_fail += 1
        items.append({
            "check_id": cdef.check_id,
            "title_ru": cdef.title_ru,
            "passed": passed,
            "weight": cdef.weight,
            "table_ref": cdef.table_ref,
            "critical": cdef.critical,
            "verdict_ru": _verdict_ru(passed, cdef.critical),
            "verdict_class": _verdict_class(passed, cdef.critical),
        })
    pct = round(100.0 * earned / total_w, 1)
    return pct, items, critical_fail


def evaluate_cisz_readiness(
    *,
    bundle: dict[str, Any] | None = None,
    text: str | None = None,
    scenario: str = "auto",
) -> dict[str, Any]:
    """Возвращает оценку готовности к ЦИСЗ и чек-лист по программе испытаний."""
    source = "none"
    results: dict[str, bool] = {}
    detected = scenario

    if bundle and isinstance(bundle, dict) and bundle.get("resourceType") == "Bundle":
        source = "fhir_bundle"
        results = inspect_bundle_checks(bundle)
        if scenario == "auto":
            detected = detect_bundle_scenario(bundle)
        include_meds = bool(results.get("medication_request"))
        if detected == "primary_ambulatory" and include_meds:
            detected = "medication"
    elif (text or "").strip():
        source = "text"
        results = inspect_text_checks(text.strip())
        detected = "primary_ambulatory" if scenario == "auto" else scenario
    else:
        return {
            "ok": False,
            "source": source,
            "scenario": detected,
            "overall_score": None,
            "overall_status": "insufficient_data",
            "critical_failures": 0,
            "checks": [],
            "summary_ru": "Нет данных для проверки готовности к ЦИСЗ.",
            "disclaimer_ru": (
                "Ориентир по программе испытаний МИС v.1.3-4; "
                "не заменяет импорт и валидацию в ЦИСЗ."
            ),
        }

    check_defs = checks_for_scenario(
        detected,
        include_medication=(detected == "medication"),
    )
    pct, items, critical_fail = _score_from_checks(check_defs, results)

    if critical_fail > 0:
        status = "non_compliant"
    elif pct >= 85:
        status = "mostly_compliant"
    elif pct >= 70:
        status = "partially_compliant"
    else:
        status = "non_compliant"

    failed = [i for i in items if not i["passed"]]
    summary = (
        f"Готовность к ЦИСЗ: {pct:.0f}% "
        f"({len(items) - len(failed)}/{len(items)} проверок, сценарий {detected})."
    )
    if critical_fail:
        summary += f" Критических пробелов: {critical_fail}."

    return {
        "ok": True,
        "source": source,
        "scenario": detected,
        "scenario_label_ru": {
            "primary_ambulatory": "Первичный приём (3.2.1)",
            "specialist_consult": "Консультация специалиста (3.13.1)",
            "medication": "Приём с лекарственным обеспечением (3.2.1 + 3.3)",
        }.get(detected, detected),
        "overall_score": pct,
        "overall_status": status,
        "critical_failures": critical_fail,
        "checks": items,
        "summary_ru": summary,
        "disclaimer_ru": (
            "Проверка по чек-листу программы испытаний МИС v.1.3-4 (амбулаторный профиль). "
            "Не заменяет метод «проверка пакета без импорта» и импорт в ЦИСЗ."
        ),
        "program_ref": "Программа испытаний МИС v.1.3-4 — Амбулаторный профиль",
    }


def merge_send_gate_with_cisz(
    send_gate: dict[str, Any],
    cisz: dict[str, Any],
    *,
    combine_scores: bool = True,
) -> dict[str, Any]:
    """Объединяет clinical gate и cisz_readiness в единый send_gate для МИС."""
    out = dict(send_gate)
    cisz_score = cisz.get("overall_score")
    out["cisz_readiness"] = cisz
    out["cisz_score"] = cisz_score
    out["clinical_gate_score"] = send_gate.get("gate_score")

    if cisz_score is not None and combine_scores:
        clinical = send_gate.get("gate_score")
        if isinstance(clinical, (int, float)):
            combined = min(float(clinical), float(cisz_score))
            out["gate_score"] = combined
            out["overall_score"] = combined
            out["score_combined"] = True
            if cisz.get("critical_failures", 0) > 0:
                prev = out.get("block_reason_ru") or ""
                cisz_note = (
                    f"Готовность к ЦИСЗ {cisz_score:.0f}%: "
                    f"критические пробелы по программе испытаний МИС."
                )
                out["block_reason_ru"] = f"{prev} {cisz_note}".strip() if prev else cisz_note
                if out.get("gate_mode") == "hard_gate":
                    out["gate_allowed"] = False
                    out["send_risk_level"] = "blocked"
        elif isinstance(cisz_score, (int, float)):
            out["gate_score"] = cisz_score
            out["overall_score"] = cisz_score
    return out


def attach_cisz_readiness(
    payload: dict[str, Any],
    *,
    bundle: dict[str, Any] | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    """Добавляет cisz_readiness и объединяет с send_gate в ответе API/пайплайна."""
    cisz = evaluate_cisz_readiness(bundle=bundle, text=text)
    payload["cisz_readiness"] = cisz
    sg = payload.get("send_gate")
    if isinstance(sg, dict):
        merged = merge_send_gate_with_cisz(sg, cisz)
        payload["send_gate"] = merged
    sa = payload.get("structured_analysis")
    if isinstance(sa, dict):
        comp = sa.get("compliance")
        if isinstance(comp, dict):
            comp["cisz_readiness"] = cisz
            if isinstance(payload.get("send_gate"), dict):
                comp["send_gate"] = payload["send_gate"]
    return payload
