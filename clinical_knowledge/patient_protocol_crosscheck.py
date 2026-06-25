"""Сверка КЗ и документов с рекомендациями клинического протокола (B2C, фаза 2c)."""
from __future__ import annotations

import re
from typing import Any

from .lab_result_parser import extract_lab_markers, marker_names


def _blob(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _mentioned(name: str, *texts: str) -> bool:
    n = _blob(name)
    if not n:
        return False
    for raw in texts:
        t = _blob(raw)
        if not t:
            continue
        if n in t:
            return True
        if len(n) <= 5 and re.search(rf"\b{re.escape(n)}\b", t, re.I):
            return True
    return False


def crosscheck_protocol_requirements(
    *,
    l1_result: dict[str, Any],
    kz_text: str,
    lab_text: str = "",
) -> dict[str, Any]:
    """Найти обязательные обследования по протоколу, отсутствующие в КЗ/анализах."""
    sa = l1_result.get("structured_analysis") if isinstance(l1_result.get("structured_analysis"), dict) else {}
    comp = sa.get("compliance") if isinstance(sa.get("compliance"), dict) else {}
    matches = sa.get("matches") if isinstance(sa.get("matches"), list) else []

    protocol_title = ""
    if matches and isinstance(matches[0], dict):
        protocol_title = str(matches[0].get("title") or "").strip()[:220]

    exam_assessments = list(comp.get("exam_assessments") or [])
    lab_names = marker_names(extract_lab_markers(lab_text)) if (lab_text or "").strip() else []

    missing: list[dict[str, Any]] = []
    for ex in exam_assessments:
        if not isinstance(ex, dict):
            continue
        status = str(ex.get("status") or "")
        if status not in ("missing_required", "missing_conditional"):
            continue
        exam_name = str(ex.get("exam_name") or "обследование").strip()
        in_kz = _mentioned(exam_name, kz_text)
        in_lab = _mentioned(exam_name, lab_text) or any(_mentioned(exam_name, ln) for ln in lab_names)
        proto_ev = [str(x).strip() for x in (ex.get("protocol_evidence") or []) if str(x).strip()]
        excerpt = proto_ev[0][:280] if proto_ev else ""
        if in_kz or in_lab:
            continue
        severity = "high" if status == "missing_required" else "medium"
        note = (
            f"По клиническому протоколу Минздрава обычно отражают «{exam_name}». "
            "В вашем заключении и загруженных анализах этого не видно - уточните у врача, нужно ли обследование."
        )
        missing.append(
            {
                "exam_name": exam_name,
                "status": status,
                "severity": severity,
                "in_kz_text": in_kz,
                "in_lab_documents": in_lab,
                "protocol_excerpt": excerpt,
                "patient_note_ru": note,
            }
        )

    notes: list[str] = []
    if missing:
        names = ", ".join(m["exam_name"] for m in missing[:4])
        if len(missing) > 4:
            names += f" и ещё {len(missing) - 4}"
        notes.append(
            f"По протоколу Минздрава для вашей ситуации могут требоваться: {names}. "
            "Проверьте, отражены ли они в заключении или назначениях."
        )
    elif exam_assessments:
        notes.append("Основные обследования из протокола в документе отражены или не требуются для этого случая.")

    return {
        "protocol_title": protocol_title,
        "missing_recommended_exams": missing[:12],
        "notes_ru": notes,
    }
