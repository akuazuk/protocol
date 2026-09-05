"""PHI-safe laboratory evidence for Endpoint C (diagnosis-to-evidence).

Warehouse labs reach the blind judge as panel labels and dates only.
Values, units, patient_id, patient_key and visit_id never enter the slot.
Primary scoring is unchanged: this module does not write exam_data and
does not grade numeric abnormality.
"""
from __future__ import annotations

from typing import Any, Mapping

ENGINE = "mo_lab_dx_evidence_v1"
SLOT = "lab"
MAX_PANELS = 8
CODE_DX_LAB_CONTEXT = "B_dx_lab_context"
NOTE_RU = (
    "Лаборатория склада для сверки диагноза: только названия панелей и даты. "
    "Значения и референс не передаются и не оцениваются."
)

_CLINICAL_KEYS = ("complaints", "anamnesis", "objective_status", "exam_data")
_DIAGNOSIS_KEYS = (
    "clinical_diagnosis",
    "diagnosis",
    "diagnosis_text",
    "diagnosis_main_text",
)


def empty_lab_evidence(*, reason: str = "") -> dict[str, Any]:
    return {
        "engine": ENGINE,
        "present": False,
        "slot": SLOT,
        "text": "",
        "panels": [],
        "n_panels": 0,
        "same_day_n": 0,
        "has_values": False,
        "reason": reason,
        "note_ru": NOTE_RU,
    }


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _visit_date(case: Mapping[str, Any] | None) -> str:
    if not isinstance(case, Mapping):
        return ""
    window = case.get("window") if isinstance(case.get("window"), Mapping) else {}
    lab = case.get("lab") if isinstance(case.get("lab"), Mapping) else {}
    lab_window = lab.get("window") if isinstance(lab.get("window"), Mapping) else {}
    return str(
        case.get("visit_date")
        or case.get("date")
        or window.get("visit_date")
        or lab_window.get("visit_date")
        or ""
    )[:10]


def _panel_bit(item: Mapping[str, Any], *, visit_date: str) -> str:
    label = str(item.get("label") or "").strip()
    test_date = str(item.get("test_date") or "")[:10]
    if not label:
        return ""
    if test_date and visit_date and test_date == visit_date:
        return f"{label} ({test_date}, день визита)"
    if test_date:
        return f"{label} ({test_date})"
    return label


def _from_panels(
    panels: list[Mapping[str, Any]],
    *,
    visit_date: str,
    reason: str = "",
) -> dict[str, Any]:
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in panels:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or "").strip()
        test_date = str(item.get("test_date") or "")[:10]
        if not label:
            continue
        key = f"{label}|{test_date}"
        if key in seen:
            continue
        seen.add(key)
        same_day = bool(item.get("same_day")) or bool(visit_date and test_date == visit_date)
        clean.append({"label": label, "test_date": test_date, "same_day": same_day})
        if len(clean) >= MAX_PANELS:
            break
    if not clean:
        return empty_lab_evidence(reason=reason or "empty")
    text = "; ".join(_panel_bit(item, visit_date=visit_date) for item in clean)
    return {
        "engine": ENGINE,
        "present": True,
        "slot": SLOT,
        "text": _clip(text, 400),
        "panels": clean,
        "n_panels": len(clean),
        "same_day_n": sum(1 for item in clean if item["same_day"]),
        "has_values": False,
        "reason": "",
        "note_ru": NOTE_RU,
    }


def _panels_from_present_map(
    present: Mapping[str, Any] | None,
    *,
    visit_date: str,
) -> list[dict[str, Any]]:
    if not isinstance(present, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for row in present.values():
        if not isinstance(row, Mapping):
            continue
        out.append(
            {
                "label": str(row.get("label") or "").strip(),
                "test_date": str(row.get("test_date") or "")[:10],
                "same_day": str(row.get("test_date") or "")[:10] == visit_date,
            }
        )
    return out


def _panels_from_reconcile(
    reconcile: Mapping[str, Any] | None,
    *,
    visit_date: str,
) -> list[dict[str, Any]]:
    if not isinstance(reconcile, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for item in reconcile.get("present") or []:
        if not isinstance(item, Mapping):
            continue
        test_date = str(item.get("test_date") or "")[:10]
        out.append(
            {
                "label": str(item.get("label") or "").strip(),
                "test_date": test_date,
                "same_day": test_date == visit_date,
            }
        )
    return out


def _has_identity(case: Mapping[str, Any]) -> bool:
    raw = case.get("raw") if isinstance(case.get("raw"), Mapping) else {}
    return bool(
        str(case.get("patient_id") or case.get("patientId") or raw.get("patient_id") or "").strip()
        or str(case.get("patient_key") or "").strip()
    )


def lab_evidence_for_dx(
    case: Mapping[str, Any] | None = None,
    *,
    bundle: Mapping[str, Any] | None = None,
    lab_db: Any = None,
) -> dict[str, Any]:
    """Build the Endpoint C lab slot from warehouse or an already-safe snippet."""
    from clinical_knowledge.mo_lab_bundle import lab_bundle_enabled

    if not lab_bundle_enabled():
        return empty_lab_evidence(reason="disabled")

    visit = _visit_date(case)
    if isinstance(case, Mapping):
        existing = case.get("lab_evidence")
        if isinstance(existing, Mapping) and (
            existing.get("present") or str(existing.get("text") or "").strip()
        ):
            panels = [
                item
                for item in (existing.get("panels") or [])
                if isinstance(item, Mapping)
            ]
            rebuilt = _from_panels(panels, visit_date=visit) if panels else empty_lab_evidence()
            text = _clip(existing.get("text"), 400)
            if text:
                rebuilt["text"] = text
                rebuilt["present"] = True
                rebuilt["n_panels"] = rebuilt["n_panels"] or 1
                rebuilt["reason"] = ""
                return rebuilt
            if rebuilt["present"]:
                return rebuilt
        lab = case.get("lab") if isinstance(case.get("lab"), Mapping) else {}
        recon_panels = _panels_from_reconcile(
            lab.get("reconcile") if isinstance(lab.get("reconcile"), Mapping) else None,
            visit_date=visit,
        )
        if recon_panels:
            return _from_panels(recon_panels, visit_date=visit)

    source = bundle
    if source is None and isinstance(case, Mapping):
        if isinstance(case.get("lab"), Mapping) and case["lab"].get("days"):
            source = case["lab"]
        elif isinstance(case.get("_lab"), Mapping):
            source = case["_lab"]
        elif lab_db is not None or _has_identity(case):
            from clinical_knowledge.mo_lab_bundle import lab_reconcile_payload_for_case

            source = lab_reconcile_payload_for_case(case, lab_db=lab_db)

    if isinstance(source, Mapping):
        from clinical_knowledge.mo_lab_shadow import present_panels

        if not visit:
            visit = str((source.get("window") or {}).get("visit_date") or "")[:10]
        present = present_panels(source, max_date=visit)
        return _from_panels(
            _panels_from_present_map(present, visit_date=visit),
            visit_date=visit,
            reason=str(source.get("reason") or "empty"),
        )
    return empty_lab_evidence(reason="empty")


def lab_evidence_text_from_source(source: Mapping[str, Any] | None) -> str:
    """Allowlisted lab text for a blind pack. Never copies indicator values."""
    if not isinstance(source, Mapping):
        return ""
    evidence = source.get("evidence") if isinstance(source.get("evidence"), Mapping) else {}
    raw_text = str(
        (source.get("lab_evidence") or {}).get("text")
        if isinstance(source.get("lab_evidence"), Mapping)
        else source.get("lab_evidence") or evidence.get("lab") or source.get("lab_evidence_text") or ""
    ).strip()
    if raw_text and not isinstance(source.get("lab"), Mapping):
        ev = lab_evidence_for_dx(
            {
                "lab_evidence": source.get("lab_evidence")
                if isinstance(source.get("lab_evidence"), Mapping)
                else {"text": raw_text, "present": True},
                "visit_date": _visit_date(source),
            }
        )
        return ev["text"]
    return lab_evidence_for_dx(source).get("text") or ""


def attach_lab_evidence_to_row(
    row: Mapping[str, Any],
    *,
    lab_db: Any = None,
) -> dict[str, Any]:
    """Put a PHI-safe snippet on the case row before blind packing."""
    out = dict(row)
    out["lab_evidence"] = lab_evidence_for_dx(out, lab_db=lab_db)
    return out


def _diagnosis_present(case: Mapping[str, Any] | None) -> bool:
    if not isinstance(case, Mapping):
        return False
    diagnosis = case.get("diagnosis")
    if isinstance(diagnosis, Mapping) and str(diagnosis.get("text") or "").strip():
        return True
    return any(str(case.get(key) or "").strip() for key in _DIAGNOSIS_KEYS)


def _clinical_text_present(case: Mapping[str, Any] | None) -> bool:
    if not isinstance(case, Mapping):
        return False
    evidence = case.get("evidence") if isinstance(case.get("evidence"), Mapping) else {}
    slots = case.get("slots") if isinstance(case.get("slots"), Mapping) else {}
    for key in _CLINICAL_KEYS:
        if any(str(container.get(key) or "").strip() for container in (case, evidence, slots)):
            return True
    return False


def lab_dx_shadow_findings(
    evidence: Mapping[str, Any] | None,
    case: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Shadow-only: diagnosis exists, doctor text is empty, warehouse labs exist."""
    if not isinstance(evidence, Mapping) or not evidence.get("present"):
        return []
    if not _diagnosis_present(case) or _clinical_text_present(case):
        return []
    bits = str(evidence.get("text") or "").strip()
    if not bits:
        return []
    return [
        {
            "code": CODE_DX_LAB_CONTEXT,
            "axis": "diagnosis",
            "severity": "P3",
            "severity_label_ru": "Оформление",
            "severity_tone": "formal",
            "passed": False,
            "title_ru": "Диагноз можно сверять с лабораторией склада",
            "detail_ru": (
                f"Клинический текст пуст, на складе есть {bits}. "
                "Значения не входят в оценку. Черновик, не входит в оценку."
            ),
            "evidence": "",
            "source_ref": ENGINE,
            "needs_human": False,
            "shadow": True,
            "is_shadow": True,
            "engine": ENGINE,
            "linked_fields": ["diagnosis", "exam_data"],
            "link_hint_ru": "Сверьте диагноз с панелями склада, не с цифрами",
        }
    ]
