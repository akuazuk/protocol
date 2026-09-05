"""Unused lab findings: результат есть, в диагнозе/плане не учтён.

План: docs/plans/2026-09-04-mo-drugs-labs-scoring-v1.md волна 1.
Primary только при MO_LAB_UNUSED_PRIMARY=1 (default off).
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from clinical_knowledge.lab_canons import (
    lab_panels,
    panels_mentioned_in_text,
    text_hits_panel,
)

ENGINE = "mo_lab_unused_v1"
_SOURCE = "mo_lab_unused_v1"
CODE_UNUSED_DX = "B_lab_unused_in_dx"
CODE_UNUSED_PLAN = "B_lab_unused_in_plan"
CODE_ORDERED_NOT_USED = "B_lab_ordered_not_used"
UNUSED_CODES = {CODE_UNUSED_DX, CODE_UNUSED_PLAN, CODE_ORDERED_NOT_USED}


def lab_unused_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_UNUSED") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def lab_unused_primary_enabled() -> bool:
    raw = (os.environ.get("MO_LAB_UNUSED_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _case_text(case: Mapping[str, Any], *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        val = case.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _present_from_exam_data(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    exam = _case_text(case, "exam_data")
    return panels_mentioned_in_text(exam)


def _present_from_reconcile(
    reconcile: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(reconcile, Mapping):
        return out
    label_to_id = {p["label"]: p["id"] for p in lab_panels()}
    for row in reconcile.get("present") or []:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("label") or "").strip()
        pid = label_to_id.get(label)
        if not pid:
            # try match by panel label case-insensitive
            for panel in lab_panels():
                if panel["label"].lower() == label.lower():
                    pid = panel["id"]
                    break
        if not pid:
            continue
        out[pid] = {
            "id": pid,
            "label": label or next(
                (p["label"] for p in lab_panels() if p["id"] == pid), pid
            ),
            "test_date": str(row.get("test_date") or "")[:10],
            "source": "warehouse",
        }
    return out


def collect_present_panels(
    case: Mapping[str, Any],
    reconcile: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    merged = _present_from_exam_data(case)
    for pid, row in _present_from_reconcile(reconcile).items():
        merged.setdefault(pid, row)
        if row.get("test_date"):
            merged[pid]["test_date"] = row["test_date"]
        merged[pid]["source"] = row.get("source") or merged[pid].get("source") or "exam_data"
    return merged


def _finding(code: str, *, title: str, detail: str, severity: str = "P1") -> dict[str, Any]:
    shadow = not lab_unused_primary_enabled()
    return {
        "code": code,
        "axis": "concordance",
        "severity": severity,
        "severity_label_ru": "Важно" if severity == "P1" else "Оформление",
        "passed": False,
        "title_ru": title,
        "detail_ru": detail
        + (" Черновик, не входит в оценку." if shadow else ""),
        "evidence": "",
        "source_ref": _SOURCE,
        "needs_human": True,
        "shadow": shadow,
        "is_shadow": shadow,
        "engine": ENGINE,
        "linked_fields": ["exam_data", "clinical_diagnosis", "treatment_recommendations"],
        "link_hint_ru": "Укажите результат анализа в диагнозе и плане",
    }


def unused_lab_findings(
    case: Mapping[str, Any] | None,
    reconcile: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not lab_unused_enabled() or not isinstance(case, Mapping):
        return []
    present = collect_present_panels(case, reconcile)
    if not present:
        return []

    dx_text = _case_text(
        case,
        "clinical_diagnosis",
        "diagnosis_main_text",
        "diagnosis",
    )
    plan_text = _case_text(
        case,
        "treatment_recommendations",
        "exam_recommendations",
        "dispensary_info",
        "return_date",
    )
    # Норма: учёт в exam_data сам по себе не закрывает unused_in_dx —
    # нужен диагноз/план. Но если панель только из warehouse и уже в exam_data,
    # для dx всё равно проверяем dx_text.

    unused_dx: list[str] = []
    unused_plan: list[str] = []
    for pid, row in present.items():
        panel = next((p for p in lab_panels() if p["id"] == pid), None)
        if not panel:
            continue
        label = str(row.get("label") or panel["label"])
        if not text_hits_panel(dx_text, panel):
            unused_dx.append(label)
        if not text_hits_panel(plan_text, panel):
            unused_plan.append(label)

    out: list[dict[str, Any]] = []
    if unused_dx:
        bits = ", ".join(unused_dx[:6])
        out.append(
            _finding(
                CODE_UNUSED_DX,
                title="Готовый анализ не учтён в диагнозе",
                detail=(
                    f"Есть результаты: {bits}, "
                    "но в диагнозе они не отражены."
                ),
            )
        )
    if unused_plan:
        bits = ", ".join(unused_plan[:6])
        out.append(
            _finding(
                CODE_UNUSED_PLAN,
                title="Готовый анализ не учтён в плане",
                detail=(
                    f"Есть результаты: {bits}, "
                    "но в плане лечения или контроля они не отражены."
                ),
            )
        )

    # Волна 3: заказ ранее / результат есть / в текущем МО не разобран —
    # приближение: present_not_in_mo из reconcile + нет в dx и plan.
    if isinstance(reconcile, Mapping):
        gap_labels = {
            str(item.get("label") or "")
            for item in (reconcile.get("present_not_in_mo") or [])
            if isinstance(item, Mapping)
        }
        ordered_not_used = [
            label
            for label in unused_dx
            if label in gap_labels and label in unused_plan
        ]
        if ordered_not_used:
            bits = ", ".join(ordered_not_used[:6])
            out.append(
                _finding(
                    CODE_ORDERED_NOT_USED,
                    title="Результат лаборатории не разобран в текущем МО",
                    detail=(
                        f"На складе есть {bits}, в диагнозе и плане визита "
                        "они не учтены."
                    ),
                )
            )
    return out


def merge_unused_into_findings(
    findings: list[dict[str, Any]] | None,
    extra: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    base = [
        dict(item)
        for item in (findings or [])
        if isinstance(item, Mapping)
        and str(item.get("code") or item.get("finding_code") or "") not in UNUSED_CODES
    ]
    for item in extra or []:
        if isinstance(item, Mapping) and item.get("code"):
            base.append(dict(item))
    return base
