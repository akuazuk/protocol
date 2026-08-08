"""Concordance findings для МО/КЗ (shadow-слой, отдельно от L1 overall).

См. docs/plans/2026-08-05-mo-eval-smirnova-concordance-v1.md.
"""
from __future__ import annotations

import os
import re
from typing import Any

from .mo_case_signals import extract_mo_case_signals

ENGINE = "mo_concordance_v1"
_SOURCE = "mo_concordance_v1"


def concordance_findings_enabled() -> bool:
    """Shadow findings. Default on; set MO_CONCORDANCE_FINDINGS=0 to disable."""
    raw = (os.environ.get("MO_CONCORDANCE_FINDINGS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def concordance_primary_enabled() -> bool:
    """Merge into primary deep findings (affects overall). Default off."""
    raw = (os.environ.get("MO_CONCORDANCE_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_LINKED: dict[str, tuple[list[str], str]] = {
    "finding_not_in_diagnosis": (
        ["objective_status", "exam_data", "clinical_diagnosis", "mkb_code_main"],
        "Сверьте находку в статусе/обследованиях с формулировкой диагноза и МКБ",
    ),
    "anamnesis_thin_for_duration": (
        ["complaints", "anamnesis_doctor", "anamnesis_auto"],
        "Сверьте длительность жалобы с полнотой анамнеза",
    ),
    "underworkup_chronic_red_flag": (
        ["complaints", "objective_status", "exam_recommendations", "treatment_recommendations"],
        "Сверьте red-flag презентацию с планом обследования и контролем",
    ),
    "plan_laterality_mismatch": (
        ["complaints", "objective_status", "treatment_recommendations"],
        "Сверьте сторону жалобы/статуса с латеральностью плана",
    ),
    "icd_weakly_supported": (
        ["clinical_diagnosis", "objective_status", "complaints"],
        "Сверьте код МКБ с клинической картиной",
    ),
    "pediatric_limp_ddx_not_addressed": (
        ["complaints", "clinical_diagnosis", "exam_recommendations"],
        "Сверьте длительную хромоту у ребёнка с DDx и планом исключения",
    ),
}


def _finding(
    code: str,
    axis: str,
    severity: str,
    *,
    title: str,
    detail: str = "",
    evidence: str = "",
) -> dict[str, Any]:
    linked, hint = _LINKED.get(code, ([], ""))
    return {
        "code": code,
        "axis": axis,
        "severity": severity,
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "engine": ENGINE,
        "linked_fields": list(linked),
        "link_hint_ru": hint,
    }


def merge_concordance_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Добавить shadow concordance в список findings case detail (без дублей по code)."""
    out = [dict(item) for item in (findings or []) if isinstance(item, dict)]
    if not concordance_findings_enabled() or not case:
        return out
    existing = {str(item.get("code") or item.get("finding_code") or "") for item in out}
    try:
        shadow = evaluate_mo_concordance(case)
    except Exception:  # noqa: BLE001
        return out
    for item in shadow:
        code = str(item.get("code") or "")
        if not code or code in existing:
            continue
        out.append({**item, "is_shadow": True})
        existing.add(code)
    # P0/P1 сначала, затем shadow concordance рядом с primary
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    out.sort(key=lambda f: (order.get(str(f.get("severity") or ""), 9), str(f.get("code") or "")))
    return out


def clinical_case_from_document(document: dict[str, Any] | None, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Собрать case dict из payload документа + record для live concordance."""
    clinical = {}
    if isinstance(document, dict):
        clinical = document.get("clinical") if isinstance(document.get("clinical"), dict) else {}
    record = record if isinstance(record, dict) else {}
    case: dict[str, Any] = {}
    for key in (
        "complaints",
        "anamnesis_doctor",
        "anamnesis_auto",
        "objective_status",
        "exam_data",
        "clinical_diagnosis",
        "mis_diagnos",
        "diagnosis_main_text",
        "diagnosis_list",
        "mkb_code_main",
        "treatment_recommendations",
        "exam_recommendations",
        "patient_age_years",
    ):
        val = clinical.get(key)
        if val in (None, "") and key in record:
            val = record.get(key)
        if val not in (None, ""):
            case[key] = val
    if "mkb_code_main" not in case and record.get("diagnosis_code"):
        case["mkb_code_main"] = record.get("diagnosis_code")
    if "patient_age_years" not in case and record.get("patient_age_years") is not None:
        case["patient_age_years"] = record.get("patient_age_years")
    return case


def evaluate_mo_concordance(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Вернуть список непройденных concordance findings (shadow shape = deep finding)."""
    sig = extract_mo_case_signals(case)
    out: list[dict[str, Any]] = []

    # B2: мягкий сигнал continuity из бандла истории (не дублирует B_patient_history_context)
    try:
        hist = case.get("_patient_history_summary")
        if isinstance(hist, dict) and str(hist.get("tier") or "") == "new_for_profile":
            prior_codes = list((hist.get("codes_same_doctor") or {}).keys())[:3]
            if prior_codes and hist.get("current_code"):
                out.append(
                    _finding(
                        "history_dx_line_break",
                        "clinical_concordance",
                        "P3",
                        title="Диагноз выбивается из линии ведения у этого врача",
                        detail=(
                            f"Текущий код {hist.get('current_code')} не встречался у врача по пациенту; "
                            f"раньше: {', '.join(prior_codes)}"
                        ),
                        evidence="",
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    # 1) finding_not_in_diagnosis
    for item in sig.get("joint_edema") or []:
        joint = str(item.get("joint") or "")
        covers = (sig.get("dx_covers_joint") or {}).get(joint)
        if joint and not covers:
            out.append(
                _finding(
                    "finding_not_in_diagnosis",
                    "clinical_concordance",
                    "P1",
                    title="Находка в статусе не отражена в диагнозе",
                    detail=f"Отёк/припухлость сустава ({joint}) есть в осмотре, но не покрыта диагнозом/МКБ",
                    evidence=str(item.get("evidence") or ""),
                )
            )

    # 2) anamnesis_thin_for_duration
    # E2: только при MSK red-flag презентации (хромота / отёк сустава).
    # Иначе ~12% FP на любой хронике (гинекология, АГ, кашель…).
    duration = sig.get("duration_days")
    red_presentation = bool(sig.get("has_limp") or sig.get("joint_edema"))
    if (
        isinstance(duration, int)
        and duration >= 28
        and red_presentation
    ):
        themes = int(sig.get("anamnesis_theme_count") or 0)
        anam_len = int(sig.get("anamnesis_len") or 0)
        # E2: themes AND short text (шаблонный длинный анамнез с 1 темой - не finding).
        if themes < 2 and anam_len < 120:
            out.append(
                _finding(
                    "anamnesis_thin_for_duration",
                    "documentation",
                    "P2",
                    title="Анамнез слишком краток для длительности жалобы",
                    detail=(
                        f"Длительность ~{duration} дн. при хромоте/отёке сустава; "
                        f"тем анамнеза {themes}/4, длина текста {anam_len} символов"
                    ),
                    evidence=(sig.get("anamnesis") or "")[:300],
                )
            )

    # 3) underworkup_chronic_red_flag
    # E2 decision: P1 только pediatric; adult → P2 (шумнее, меньше safety-сигнал).
    pediatric = sig.get("audience") == "pediatric"
    chronic = isinstance(duration, int) and duration >= 28
    no_workup = not (sig.get("plan_has_imaging") or sig.get("plan_has_labs"))
    worsening_only = bool(sig.get("follow_up_on_worsening_only"))
    if chronic and red_presentation and no_workup and worsening_only:
        sev = "P1" if pediatric else "P2"
        out.append(
            _finding(
                "underworkup_chronic_red_flag",
                "safety",
                sev,
                title="Недостаточный объём обследования при хроническом сценарии",
                detail=(
                    "Длительные жалобы с хромотой/отёком сустава при отсутствии "
                    "imaging/labs в плане и контроле только при ухудшении"
                ),
                evidence=(sig.get("plan") or "")[:300],
            )
        )

    # 4) plan_laterality_mismatch
    side = sig.get("complaint_side")
    if side in {"right", "left"} and sig.get("plan_bilateral"):
        out.append(
            _finding(
                "plan_laterality_mismatch",
                "clinical_concordance",
                "P3",
                title="Латеральность плана не совпадает с жалобой",
                detail="Жалоба/статус односторонние, в плане двусторонняя процедура без обоснования",
                evidence=f"side={side}; plan={(sig.get('plan') or '')[:220]}",
            )
        )

    # 5) icd_weakly_supported (start: M60*)
    icd = str(sig.get("icd") or "")
    if re.match(r"^M60", icd) and not sig.get("infection_signs"):
        out.append(
            _finding(
                "icd_weakly_supported",
                "clinical_concordance",
                "P2",
                title="Код МКБ слабо поддержан клинической картиной",
                detail="M60* при отсутствии признаков инфекции/системного воспаления в тексте",
                evidence=f"ICD={icd}; dx={(sig.get('diagnosis') or '')[:200]}",
            )
        )

    # 6) pediatric_limp_ddx_not_addressed
    if (
        pediatric
        and sig.get("has_limp")
        and isinstance(duration, int)
        and duration >= 28
        and not sig.get("ped_limp_dx_ok")
    ):
        out.append(
            _finding(
                "pediatric_limp_ddx_not_addressed",
                "clinical_concordance",
                "P2",
                title="Не закрыт детский DDx длительной хромоты",
                detail="Нет диагноза/плана из ожидаемого ряда (ЮА, Пертес, травма, инфекция сустава и т.п.)",
                evidence=f"dx={(sig.get('diagnosis') or '')[:200]}; icd={icd}",
            )
        )

    return out
