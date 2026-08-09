"""Итог разбора случая МО (machine brief для UI).

См. docs/plans/2026-08-09-mo-case-review-quality-parity-v1.md.
Не зависит от LLM. Собирает зоны, критерии, gaps, МКБ, КП, doctor_feedback.
"""
from __future__ import annotations

from typing import Any

ENGINE = "mo_case_review_brief_v1"
_MAX_FEEDBACK = 6
# Клинические коды, которые важнее «шума» при лимите пунктов врачу.
_FEEDBACK_PRIORITY_CODES = {
    "B_complaint_exam_mismatch": 0,
    "B_chronic_dx_therapy_absent": 0,
    "B_dx_not_in_exam": 1,
    "B_tentative_dx_weak_support": 1,
    "B_treatment_before_confirmed_dx": 2,
    "B_complaint_not_addressed_in_plan": 2,
    "finding_not_in_diagnosis": 1,
    "icd_weakly_supported": 1,
    "A_text_noise": 3,
}

_BAND_RU = {
    "ok": "в норме",
    "weak": "слабо",
    "bad": "плохо",
    "na": "нет данных",
    "unmatched": "протокол не подобран",
}

_ICD_STATUS_RU = {
    "ok": "МКБ ✓ - формулировка/код согласуются со справочником",
    "missing_dx": "нет Dx - в МО нет формулировки диагноза и кода",
    "not_in_directory": "не в МКБ - формулировка или код не найдены в справочнике",
    "weak_name": "слабо МКБ - название слабо совпадает со справочником (не зона «Диагноз»)",
    "unknown": "МКБ ? - оценка ещё не посчитана",
}


def _band_label(band: str | None) -> str:
    return _BAND_RU.get(str(band or ""), str(band or "-"))


def _zone_why(zones: dict[str, Any], zone_key: str, zone_id: str) -> str:
    z = zones.get(zone_key) if isinstance(zones.get(zone_key), dict) else {}
    band = str(z.get("band") or "na")
    if band == "na" and z.get("kp_status") == "unmatched":
        return "Протокол не подобран - критерии плана не штрафуем"
    weak: list[str] = []
    for c in zones.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("zone") or "") != zone_id:
            continue
        score = c.get("score")
        if score == 0 or score == 0.5 or (score is None and c.get("na_reason")):
            title = str(c.get("title") or c.get("id") or "")
            reason = str(c.get("reason") or "")
            mark = "0" if score == 0 else ("0.5" if score == 0.5 else "н/д")
            weak.append(f"{title} ({mark})" + (f" - {reason}" if reason else ""))
    if weak:
        # Явно: почему band слабо при высоком %
        pct = z.get("pct")
        head = f"есть критерий {weak[0]}"
        if pct is not None and band == "weak":
            return f"{head} (итого {pct}% - полоса «слабо» из-за 0.5/0)"
        return head
    label = z.get("label_ru") or zone_key
    return f"{label}: {_band_label(band)}"


def _diagnosis_axes(
    zones: dict[str, Any],
    icd_status: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Три оси диагноза - не смешивать в UI."""
    z2 = zones.get("zone2a") if isinstance(zones.get("zone2a"), dict) else {}
    method = {
        "label_ru": "Методика: диагноз + МКБ в МО",
        "band": z2.get("band") or "na",
        "band_ru": _band_label(z2.get("band")),
        "detail_ru": _zone_why(zones, "zone2a", "diagnosis"),
    }
    st = ""
    if isinstance(icd_status, dict):
        st = str(icd_status.get("status") or "")
    icd_axis = {
        "label_ru": "МКБ-справочник",
        "status": st or "unknown",
        "detail_ru": _ICD_STATUS_RU.get(st, _ICD_STATUS_RU["unknown"]),
        "note_ru": "Это не зона «Диагноз» методики - отдельная проверка справочника.",
    }
    gap_codes = {
        "B_complaint_exam_mismatch",
        "B_dx_not_in_exam",
        "B_tentative_dx_weak_support",
        "B_chronic_dx_therapy_absent",
        "finding_not_in_diagnosis",
        "icd_weakly_supported",
    }
    clinical_hits = [
        f
        for f in findings
        if isinstance(f, dict) and str(f.get("code") or "") in gap_codes
    ]
    if clinical_hits:
        support = {
            "label_ru": "Клиническая опора диагноза",
            "band": "weak",
            "band_ru": "слабо",
            "detail_ru": str(clinical_hits[0].get("title_ru") or clinical_hits[0].get("detail_ru") or ""),
            "n": len(clinical_hits),
        }
    else:
        support = {
            "label_ru": "Клиническая опора диагноза",
            "band": "ok",
            "band_ru": "в норме",
            "detail_ru": "Явных разрывов клиника↔диагноз не найдено (machine).",
            "n": 0,
        }
    return {"methodology": method, "icd_directory": icd_axis, "clinical_support": support}


def _protocol_line(suggest: dict[str, Any] | None, zones: dict[str, Any]) -> dict[str, Any]:
    z2b = zones.get("zone2b") if isinstance(zones.get("zone2b"), dict) else {}
    kp = str(z2b.get("kp_status") or "")
    if not suggest or not suggest.get("available"):
        return {
            "matched": False,
            "kp_status": kp or "unmatched",
            "title": None,
            "detail_ru": "Протокол не подобран - план не штрафуем за несоответствие протоколу.",
            "items_n": 0,
        }
    items = suggest.get("items") if isinstance(suggest.get("items"), list) else []
    top = items[0] if items else {}
    title = str(top.get("title") or "") if isinstance(top, dict) else ""
    more = max(0, len(items) - 1)
    if title:
        detail = f"Топ-1: {title}" + (f" · ещё {more}" if more else "")
    else:
        detail = "Протоколы подобраны, название уточните в блоке КП"
    return {
        "matched": True,
        "kp_status": kp or "matched",
        "title": title or None,
        "protocol_id": (top.get("protocol_id") if isinstance(top, dict) else None),
        "detail_ru": detail,
        "items_n": len(items),
        "secondary_ru": (
            "При мультидиагнозе смежный КП смотрите в «ещё N»."
            if more
            else None
        ),
    }


def _history_line(bundle: dict[str, Any] | None) -> str:
    if not isinstance(bundle, dict):
        return "Нет prior - коррекции плана не оцениваются."
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    n = int(summary.get("n_visits") or 0)
    same_doc = len(bundle.get("same_doctor") or [])
    same_spec = len(bundle.get("same_specialty") or [])
    prior = "есть prior" if n > 0 else "нет prior"
    return f"К этому врачу: {same_doc} · К специальности: {same_spec} · Всего: {n} · Для коррекций: {prior}"


def synthesize_doctor_feedback(
    *,
    zones: dict[str, Any],
    findings: list[dict[str, Any]],
    protocol: dict[str, Any],
    icd_status: dict[str, Any] | None,
    history_line: str,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()

    def add(text: Any, *, priority: int) -> None:
        t = " ".join(str(text or "").split())
        if len(t) < 8:
            return
        key = t.lower()[:80]
        if key in seen:
            return
        seen.add(key)
        scored.append((priority, t))

    for f in findings:
        if not isinstance(f, dict):
            continue
        if str(f.get("severity") or "") == "P0" or str(f.get("axis") or "") == "safety":
            add(f.get("title_ru") or f.get("detail_ru") or f.get("code"), priority=0)

    for f in findings:
        if not isinstance(f, dict):
            continue
        code = str(f.get("code") or "")
        sev = str(f.get("severity") or "")
        axis = str(f.get("axis") or "")
        if sev not in {"P0", "P1", "P2", "P3"} and not code.startswith(("B_", "A_")):
            continue
        if code.startswith(("B_", "A_", "finding_")) or "concordance" in axis:
            title = str(f.get("title_ru") or "")
            detail = str(f.get("detail_ru") or "")
            # 10+code_rank: клиника раньше оформления/шума, но после P0 (priority 0)
            code_rank = _FEEDBACK_PRIORITY_CODES.get(code, 4)
            add(
                f"{title}: {detail}" if detail and detail not in title else title,
                priority=10 + code_rank,
            )

    for c in zones.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        if c.get("score") not in (0, 0.5):
            continue
        title = str(c.get("title") or c.get("id") or "")
        reason = str(c.get("reason") or "")
        add(f"Дополнить: {title}" + (f" - {reason}" if reason else ""), priority=20)

    st = str((icd_status or {}).get("status") or "")
    if st in {"missing_dx", "not_in_directory", "weak_name"}:
        add("Уточнить формулировку и/или код МКБ по справочнику", priority=12)

    # КП / план - сразу после safety (priority 0), до клинического шума (10+).
    if not protocol.get("matched") or protocol.get("kp_status") == "unmatched":
        add(
            "КП не подобран - план не штрафуем; проверьте подбор протокола по диагнозу",
            priority=5,
        )
    else:
        z2b = zones.get("zone2b") if isinstance(zones.get("zone2b"), dict) else {}
        if z2b.get("band") in {"weak", "bad"}:
            add(
                f"Сверить план с КП: {protocol.get('title') or 'подобранный протокол'}",
                priority=5,
            )

    if "нет prior" in (history_line or "").lower():
        add("Коррекции плана не оценивались (первый контакт / нет prior)", priority=25)

    scored.sort(key=lambda item: (item[0], item[1]))
    return [text for _, text in scored[:_MAX_FEEDBACK]]


def build_case_review_brief(case_detail: dict[str, Any] | None) -> dict[str, Any]:
    """Собрать review_brief из payload case detail."""
    data = case_detail if isinstance(case_detail, dict) else {}
    zones = data.get("zones") if isinstance(data.get("zones"), dict) else {}
    findings = [f for f in (data.get("findings") or []) if isinstance(f, dict)]
    icd_status = data.get("icd_visit_status") if isinstance(data.get("icd_visit_status"), dict) else None
    suggest = data.get("protocol_suggest") if isinstance(data.get("protocol_suggest"), dict) else None
    history = data.get("patient_history") if isinstance(data.get("patient_history"), dict) else None
    record = data.get("record") if isinstance(data.get("record"), dict) else {}
    narrative = data.get("case_narrative") if isinstance(data.get("case_narrative"), dict) else None

    if not zones.get("ok"):
        return {
            "ok": False,
            "engine": ENGINE,
            "available": False,
            "reason": "Зоны не посчитаны - итог разбора недоступен",
        }

    methodology_weak = []
    for c in zones.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        if c.get("score") in (0, 0.5) or (c.get("score") is None and c.get("na_reason")):
            methodology_weak.append(
                {
                    "id": c.get("id"),
                    "title": c.get("title") or c.get("id"),
                    "score": c.get("score"),
                    "score_label": c.get("score_label"),
                    "reason": c.get("reason"),
                    "zone": c.get("zone"),
                    "evidence": c.get("evidence"),
                }
            )

    gap_codes_prefix = ("B_", "finding_", "A_text_noise", "history_")
    clinical_gaps = []
    for f in findings:
        code = str(f.get("code") or "")
        if not code.startswith(gap_codes_prefix) and str(f.get("axis") or "") not in {
            "clinical_concordance",
            "icd_name_match",
            "icd_directory",
        }:
            # keep clinical gaps + ICD name for brief; skip pure safety elsewhere? include P0/P1
            if str(f.get("severity") or "") not in {"P0", "P1"}:
                continue
        if code.startswith("B_icd_") or code.startswith("B_dx_absent"):
            continue  # МКБ - отдельная строка
        clinical_gaps.append(
            {
                "code": code,
                "title_ru": f.get("title_ru") or f.get("title") or code,
                "detail_ru": f.get("detail_ru") or f.get("detail") or "",
                "severity": f.get("severity"),
                "linked_fields": f.get("linked_fields") or [],
                "evidence": f.get("evidence") or "",
            }
        )

    protocol = _protocol_line(suggest, zones)
    history_line = _history_line(history)
    diagnosis_axes = _diagnosis_axes(zones, icd_status, findings)
    doctor_feedback = synthesize_doctor_feedback(
        zones=zones,
        findings=findings,
        protocol=protocol,
        icd_status=icd_status,
        history_line=history_line,
    )
    # LLM narrative может дополнить feedback, не заменяя machine
    if narrative and narrative.get("available") and isinstance(narrative.get("doctor_feedback_ru"), list):
        for line in narrative["doctor_feedback_ru"][:3]:
            if line and str(line) not in doctor_feedback and len(doctor_feedback) < _MAX_FEEDBACK:
                doctor_feedback.append(f"[ИИ] {line}")

    zones_summary = {
        "documentation": {
            "band": (zones.get("zone1") or {}).get("band"),
            "band_ru": _band_label((zones.get("zone1") or {}).get("band")),
            "pct": (zones.get("zone1") or {}).get("pct"),
            "why_ru": _zone_why(zones, "zone1", "documentation"),
        },
        "diagnosis": {
            "band": (zones.get("zone2a") or {}).get("band"),
            "band_ru": _band_label((zones.get("zone2a") or {}).get("band")),
            "pct": (zones.get("zone2a") or {}).get("pct"),
            "why_ru": _zone_why(zones, "zone2a", "diagnosis"),
        },
        "plan": {
            "band": (zones.get("zone2b") or {}).get("band"),
            "band_ru": _band_label((zones.get("zone2b") or {}).get("band")),
            "pct": (zones.get("zone2b") or {}).get("pct"),
            "kp_status": (zones.get("zone2b") or {}).get("kp_status"),
            "why_ru": _zone_why(zones, "zone2b", "plan"),
        },
        "safety": zones.get("safety") if isinstance(zones.get("safety"), dict) else {"band": "none"},
    }

    summary_ru = "; ".join(
        [
            f"Оформление - {zones_summary['documentation']['band_ru']}",
            f"Диагноз - {zones_summary['diagnosis']['band_ru']}",
            f"План - {zones_summary['plan']['band_ru']}",
        ]
    )

    return {
        "ok": True,
        "available": True,
        "engine": ENGINE,
        "summary_ru": summary_ru,
        "header": {
            "visit_id": record.get("visit_id") or record.get("mis_id"),
            "patient_id": record.get("patient_id"),
            "date": record.get("date") or record.get("visit_date"),
            "doctor": record.get("doctor_name") or record.get("specialist_name"),
            "specialty": record.get("specialization") or record.get("specialty"),
            "diagnosis": record.get("diagnosis_text") or record.get("clinical_diagnosis"),
            "mkb": record.get("diagnosis_code") or record.get("mkb_code_main"),
        },
        "zones": zones_summary,
        "methodology_weak": methodology_weak[:12],
        "diagnosis_axes": diagnosis_axes,
        "clinical_gaps": clinical_gaps[:10],
        "icd": {
            "status": (icd_status or {}).get("status"),
            "label_ru": (icd_status or {}).get("label_ru")
            or (icd_status or {}).get("status_label_ru"),
            "detail_ru": _ICD_STATUS_RU.get(
                str((icd_status or {}).get("status") or ""),
                _ICD_STATUS_RU["unknown"],
            ),
            "note_ru": "Не путать с зоной «Диагноз» методики.",
        },
        "protocol": protocol,
        "history_ru": history_line,
        "doctor_feedback": doctor_feedback,
        "decision_summary_ru": "\n".join(f"• {b}" for b in doctor_feedback),
        "confidence": {
            "machine_ru": "Зоны, критерии, МКБ-чип и шаблоны «врачу» - детерминированно.",
            "ai_ru": (
                "Есть черновик ИИ (не меняет зоны)."
                if narrative and narrative.get("available")
                else "Черновик ИИ не загружен."
            ),
        },
    }
