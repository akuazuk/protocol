"""Клинические разрывы МО для разбора случая (shadow).

См. docs/plans/2026-08-09-mo-case-review-quality-parity-v1.md фаза C + §11.2.
Дополняет mo_concordance_findings (Смирнова), не заменяет.
"""
from __future__ import annotations

import os
import re
from typing import Any

ENGINE = "mo_clinical_gaps_v1"
_SOURCE = "mo_clinical_gaps_v1"

# Жалоба → маркеры отрицания в осмотре
_COMPLAINT_EXAM_AXES: list[tuple[str, re.Pattern[str], list[re.Pattern[str]]]] = [
    (
        "кашель",
        re.compile(r"каше?л", re.I),
        [
            re.compile(r"каше?л[аяюеи]?\s+нет", re.I),
            re.compile(r"без\s+каше?л", re.I),
        ],
    ),
    (
        "насморк / нос",
        re.compile(r"насморк|риноре|заложенн\w*\s+нос", re.I),
        [
            re.compile(r"носов\w*\s+дыхани\w*\s+не\s+затрудн", re.I),
            re.compile(r"нос\w*\s+свободн", re.I),
        ],
    ),
    (
        "головная боль",
        re.compile(r"головн\w*\s+бол", re.I),
        [re.compile(r"головн\w*\s+бол\w*\s+нет", re.I)],
    ),
]

_DX_ORGAN_MARKERS: list[tuple[str, re.Pattern[str], list[re.Pattern[str]]]] = [
    (
        "стопы / ортопедия",
        re.compile(r"стоп|плоско.?вальг|pes\s*plan|M21", re.I),
        [
            re.compile(r"стоп", re.I),
            re.compile(r"плоско", re.I),
            re.compile(r"вальг", re.I),
            re.compile(r"ортопед", re.I),
        ],
    ),
    (
        "ногти",
        re.compile(r"ногт|борозд", re.I),
        [re.compile(r"ногт", re.I), re.compile(r"борозд", re.I)],
    ),
    (
        "головная боль / неврология",
        re.compile(r"головн\w*\s+бол|G44|напряжен", re.I),
        [
            re.compile(r"невролог", re.I),
            re.compile(r"менингеал", re.I),
            re.compile(r"очагов", re.I),
            re.compile(r"черепн", re.I),
        ],
    ),
]

_CHRONIC_DX = re.compile(
    r"бронхиальн\w*\s+астм|астм[аые]|J45|сахарн\w*\s+диабет|гипертоническ\w*\s+болезн",
    re.I,
)
_THERAPY_MARKERS = re.compile(
    r"ингалят|базисн|симбикорт|серетид|сальбутамол|вентолин|"
    r"терапия|принимает|получает|контроль\w*\s+астм|ACT\b|GINA",
    re.I,
)
_TEXT_NOISE = re.compile(
    r"\bped\s+at\s+scab\b|"
    r"\bscab\s+abs\b|"
    r"[А-Яа-яЁё]\s*\d{2}\s*\.\s*\d|"  # кириллическая буква перед кодом
    r"перенсене|"
    r"xxx\s*\.?\s*\d*",
    re.I,
)
_WEIGHT_COMPLAINT = re.compile(r"набор\s+вес|прибавк\w*\s+вес|ожирен|питани", re.I)
_WEIGHT_PLAN = re.compile(r"диет|нутриц|эндокрин|ИМТ|вес|питани", re.I)
_TENTATIVE_DX = re.compile(
    r"([A-Za-zА-Яа-я]\s?\d{2}(?:\.\d{1,4})?\s*[^?.!]{0,80}\?|"
    r"[^.]{8,80}\?)",
    re.I,
)


def clinical_gaps_enabled() -> bool:
    raw = (os.environ.get("MO_CLINICAL_GAPS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _finding(
    code: str,
    *,
    severity: str,
    title: str,
    detail: str = "",
    evidence: str = "",
    linked: list[str] | None = None,
    hint: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "axis": "clinical_concordance",
        "severity": severity,
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "engine": ENGINE,
        "linked_fields": list(linked or []),
        "link_hint_ru": hint or "Сверьте связанные поля МО",
    }


def _slot(case: dict[str, Any], *keys: str) -> str:
    return " ".join(str(case.get(k) or "") for k in keys).strip()


def evaluate_mo_clinical_gaps(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not clinical_gaps_enabled() or not isinstance(case, dict):
        return []
    complaints = _slot(case, "complaints")
    exam = _slot(case, "objective_status")
    dx = _slot(case, "clinical_diagnosis", "mis_diagnos", "diagnosis_main_text")
    anamnesis = _slot(case, "anamnesis_doctor", "anamnesis_auto")
    plan = _slot(case, "exam_recommendations", "treatment_recommendations")
    treatment = _slot(case, "treatment_recommendations")
    out: list[dict[str, Any]] = []

    for label, complaint_re, negations in _COMPLAINT_EXAM_AXES:
        if not complaints or not exam:
            break
        if not complaint_re.search(complaints):
            continue
        if any(n.search(exam) for n in negations):
            out.append(
                _finding(
                    "B_complaint_exam_mismatch",
                    severity="P2",
                    title="Жалоба не согласуется с осмотром",
                    detail=f"В жалобах есть «{label}», в осмотре - явное отрицание той же оси.",
                    evidence=(complaints[:120] + " | " + exam[:160])[:400],
                    linked=["complaints", "objective_status"],
                    hint="Согласуйте описание осмотра с жалобами",
                )
            )
            break  # один finding на кейс достаточно

    for label, dx_re, exam_markers in _DX_ORGAN_MARKERS:
        if not dx or not exam:
            break
        if not dx_re.search(dx) and not (complaints and dx_re.search(complaints)):
            continue
        if any(m.search(exam) for m in exam_markers):
            continue
        # «без отклонений» по КОС при ортопедическом Dx
        generic_ok = re.search(r"костн|мышечн|скелет|ортопед", exam, re.I) and re.search(
            r"без\s+отклонен|норма|N\b", exam, re.I
        )
        if label.startswith("стопы") and generic_ok:
            out.append(
                _finding(
                    "B_dx_not_in_exam",
                    severity="P2",
                    title="Диагноз не отражён в осмотре",
                    detail=f"В диагнозе/жалобах есть {label}, осмотр не описывает находку (есть общая формулировка без деталей).",
                    evidence=(dx[:160] + " | " + exam[:160])[:400],
                    linked=["clinical_diagnosis", "objective_status"],
                    hint="Опишите локальный статус или снимите предположительный диагноз",
                )
            )
            break
        if not any(m.search(exam) for m in exam_markers):
            out.append(
                _finding(
                    "B_dx_not_in_exam",
                    severity="P2",
                    title="Диагноз не отражён в осмотре",
                    detail=f"Тема «{label}» есть в диагнозе/жалобах, в осмотре целенаправленно не описана.",
                    evidence=(dx[:160] + " | " + exam[:120])[:400],
                    linked=["clinical_diagnosis", "objective_status", "complaints"],
                )
            )
            break

    if dx and "?" in dx:
        # tentative без опоры в статусе/обследованиях
        tentative_bits = [m.group(0).strip() for m in _TENTATIVE_DX.finditer(dx)]
        weak = []
        for bit in tentative_bits[:4]:
            tokens = [t for t in re.findall(r"[а-яёa-z]{4,}", bit.lower()) if t not in {"диагноз", "неуточненный"}]
            blob = f"{exam} {_slot(case, 'exam_data')}".lower()
            if tokens and not any(t in blob for t in tokens[:3]):
                weak.append(bit[:80])
        if weak:
            out.append(
                _finding(
                    "B_tentative_dx_weak_support",
                    severity="P2",
                    title="Предположительный диагноз слабо поддержан осмотром",
                    detail="Есть формулировки с «?» без явной опоры в статусе/обследованиях: "
                    + "; ".join(weak[:3]),
                    evidence=dx[:220],
                    linked=["clinical_diagnosis", "objective_status", "exam_data"],
                )
            )

    chronic_blob = f"{dx} {anamnesis}"
    if _CHRONIC_DX.search(chronic_blob) and not _THERAPY_MARKERS.search(
        f"{anamnesis} {treatment} {dx}"
    ):
        out.append(
            _finding(
                "B_chronic_dx_therapy_absent",
                severity="P2",
                title="Хронический диагноз без описания текущей терапии",
                detail="В анамнезе/диагнозе есть хроническое заболевание (напр. астма), "
                "но не видно текущей базисной терапии или контроля.",
                evidence=chronic_blob[:220],
                linked=["anamnesis_doctor", "clinical_diagnosis", "treatment_recommendations"],
                hint="Укажите текущую терапию и контроль заболевания",
            )
        )

    noise_blob = f"{dx} {exam} {anamnesis} {complaints}"
    noise_hit = _TEXT_NOISE.search(noise_blob)
    if noise_hit:
        out.append(
            _finding(
                "A_text_noise",
                severity="P3",
                title="В тексте МО есть опечатки или мусор OCR",
                detail="Найдены артефакты оформления (латиница-обрывки, кириллица в коде МКБ, грубые опечатки).",
                evidence=noise_hit.group(0)[:80],
                linked=["clinical_diagnosis", "objective_status", "anamnesis_doctor"],
                hint="Почистите текст перед сохранением",
            )
        )

    if treatment and re.search(r"витамин\s*d|витамин\s*д|colecalciferol|холекальциферол", treatment, re.I):
        if re.search(r"E\s?55|дефицит\s+витамин\w*\s*д", dx, re.I) and "?" in dx:
            out.append(
                _finding(
                    "B_treatment_before_confirmed_dx",
                    severity="P3",
                    title="Лечение назначено при неподтверждённом диагнозе",
                    detail="Витамин D назначен при предположительном дефиците (Dx с «?») до лабораторного подтверждения.",
                    evidence=(dx[:120] + " | " + treatment[:120])[:400],
                    linked=["clinical_diagnosis", "treatment_recommendations"],
                )
            )

    if complaints and plan and _WEIGHT_COMPLAINT.search(complaints) and not _WEIGHT_PLAN.search(plan):
        out.append(
            _finding(
                "B_complaint_not_addressed_in_plan",
                severity="P3",
                title="Жалоба не закрыта планом",
                detail="В жалобах есть тема веса/питания, в рекомендациях маршрута по этому поводу нет.",
                evidence=complaints[:200],
                linked=["complaints", "exam_recommendations", "treatment_recommendations"],
            )
        )

    return out


def merge_clinical_gaps_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out = [dict(item) for item in (findings or []) if isinstance(item, dict)]
    if not clinical_gaps_enabled() or not case:
        return out
    existing = {str(item.get("code") or item.get("finding_code") or "") for item in out}
    try:
        shadow = evaluate_mo_clinical_gaps(case)
    except Exception:  # noqa: BLE001
        return out
    for item in shadow:
        code = str(item.get("code") or "")
        if not code or code in existing:
            continue
        out.append({**item, "is_shadow": True})
        existing.add(code)
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    out.sort(key=lambda f: (order.get(str(f.get("severity") or ""), 9), str(f.get("code") or "")))
    return out
