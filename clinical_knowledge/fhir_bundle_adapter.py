"""Адаптер FHIR BY Bundle → карточка КЗ и синтетический текст для Protocol.

Профили по примеру fhir.by (Bundle/MedicationDocument, Patient, Encounter,
Condition/FinalDiagnosis, Observation/VitalSigns, Composition/CompDocument).
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from .consult_parser import parse_consultation
from .consult_schema import ConsultationDocument


def _resources_by_type(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ent in bundle.get("entry") or []:
        if not isinstance(ent, dict):
            continue
        res = ent.get("resource")
        if not isinstance(res, dict):
            continue
        rt = str(res.get("resourceType") or "")
        out.setdefault(rt, []).append(res)
    return out


def _patient_gender(raw: str | None) -> str:
    g = (raw or "").lower()
    if g in ("male", "female"):
        return g
    return "unknown"


def _icd_from_condition(cond: dict[str, Any]) -> tuple[str | None, str]:
    code_obj = cond.get("code") or {}
    icd = None
    for c in code_obj.get("coding") or []:
        if not isinstance(c, dict):
            continue
        sys = str(c.get("system") or "")
        if "InternClassificDiseases" in sys or "icd" in sys.lower():
            icd = str(c.get("code") or "").upper().strip() or None
            break
    text = str(code_obj.get("text") or "").strip()
    return icd, text


def _observation_value(obs: dict[str, Any]) -> str | None:
    vq = obs.get("valueQuantity")
    if isinstance(vq, dict) and vq.get("value") is not None:
        unit = str(vq.get("unit") or vq.get("code") or "")
        return f"{vq['value']} {unit}".strip()
    comps = obs.get("component") or []
    parts: list[str] = []
    for c in comps:
        if not isinstance(c, dict):
            continue
        cv = c.get("valueQuantity")
        if isinstance(cv, dict) and cv.get("value") is not None:
            code = ""
            cod = c.get("code") or {}
            for cc in cod.get("coding") or []:
                if isinstance(cc, dict) and cc.get("code"):
                    code = str(cc["code"])
                    break
            parts.append(f"{code or 'значение'}={cv['value']}")
    return "; ".join(parts) if parts else None


def _obs_code(obs: dict[str, Any]) -> str:
    cod = obs.get("code") or {}
    for c in cod.get("coding") or []:
        if isinstance(c, dict) and c.get("code"):
            return str(c["code"])
    return ""


def bundle_to_consultation_document(
    bundle: dict[str, Any],
    *,
    consultation_id: str = "fhir",
    source_file: str = "fhir_bundle.json",
) -> ConsultationDocument:
    """Структурированная карточка КЗ из FHIR BY Bundle."""
    by_type = _resources_by_type(bundle)
    patients = by_type.get("Patient") or [{}]
    patient = patients[0] if patients else {}
    encounters = by_type.get("Encounter") or []
    encounter = encounters[0] if encounters else {}
    conditions = by_type.get("Condition") or []
    observations = by_type.get("Observation") or []
    compositions = by_type.get("Composition") or []
    practitioners = by_type.get("Practitioner") or []
    pract_roles = by_type.get("PractitionerRole") or []

    # Пациент
    names = patient.get("name") or []
    family = given = ""
    if names and isinstance(names[0], dict):
        family = str(names[0].get("family") or "")
        gv = names[0].get("given") or []
        given = " ".join(str(x) for x in gv if x)
    full_name = " ".join(x for x in (family, given) if x).strip() or None
    birth_raw = patient.get("birthDate")
    birth_date = None
    if birth_raw:
        try:
            birth_date = _dt.date.fromisoformat(str(birth_raw)[:10])
        except ValueError:
            birth_date = None

    # Врач
    doctor_name = None
    if practitioners:
        pn = (practitioners[0].get("name") or [{}])[0] if practitioners[0].get("name") else {}
        if isinstance(pn, dict):
            doctor_name = " ".join(
                x for x in (pn.get("family"), " ".join(pn.get("given") or [])) if x
            ).strip() or None
    if not doctor_name and pract_roles:
        doctor_name = str(
            ((encounter.get("participant") or [{}])[0].get("actor") or {}).get("display") or ""
        ).strip() or None

    # Дата визита
    consult_date = None
    period = encounter.get("actualPeriod") or {}
    start = period.get("start") or encounter.get("period", {}).get("start")
    if compositions and compositions[0].get("date"):
        start = compositions[0].get("date")
    if start:
        try:
            consult_date = _dt.date.fromisoformat(str(start)[:10])
        except ValueError:
            consult_date = None

    # Диагнозы
    diag_lines: list[str] = []
    for cond in conditions:
        icd, text = _icd_from_condition(cond)
        line = f"{icd + ' ' if icd else ''}{text}".strip()
        if line:
            diag_lines.append(line)

    # Наблюдения
    vitals: dict[str, str] = {}
    obj_parts: list[str] = []
    _OBS_RU = {
        "body-height": "Рост",
        "body-weight": "Вес",
        "body-mass-index": "ИМТ",
        "arterial-blood-pressure": "АД",
        "heart-rate": "ЧСС",
        "body-temperature": "Температура",
    }
    for obs in observations:
        code = _obs_code(obs)
        val = _observation_value(obs)
        if not val:
            continue
        label = _OBS_RU.get(code, code)
        obj_parts.append(f"{label}: {val}")
        vitals[code or label] = val

    text = _synthesize_bundle_text(bundle, doc_diagnoses=diag_lines, vitals=vitals, doctor_name=doctor_name)
    doc = parse_consultation(
        text,
        consultation_id=consultation_id,
        source_file=source_file,
        source_file_type="fhir_bundle",
    )
    # Перезапись точных полей из FHIR поверх эвристик парсера
    if full_name:
        doc.patient.full_name = full_name
    if birth_date:
        doc.patient.birth_date = birth_date
    if patient.get("gender"):
        doc.patient.sex = _patient_gender(str(patient.get("gender")))
    if consult_date:
        doc.consultation_date = consult_date
    if doctor_name:
        doc.doctor_name = doctor_name
    if vitals:
        doc.patient.vitals.update(vitals)
    if diag_lines and not doc.diagnoses:
        from .consult_parser import parse_diagnoses

        doc.diagnoses = parse_diagnoses("; ".join(diag_lines))
    doc.source_file_type = "fhir_bundle"
    return doc


def _synthesize_bundle_text(
    bundle: dict[str, Any],
    *,
    doc_diagnoses: list[str] | None = None,
    vitals: dict[str, str] | None = None,
    doctor_name: str | None = None,
) -> str:
    by_type = _resources_by_type(bundle)
    patients = by_type.get("Patient") or [{}]
    patient = patients[0] if patients else {}
    encounters = by_type.get("Encounter") or []
    encounter = encounters[0] if encounters else {}
    compositions = by_type.get("Composition") or []

    lines: list[str] = []
    if doctor_name:
        lines.append(f"Врач: {doctor_name}")

    consult_date = None
    period = encounter.get("actualPeriod") or {}
    start = period.get("start")
    if compositions and compositions[0].get("date"):
        start = compositions[0].get("date")
    if start:
        try:
            consult_date = _dt.date.fromisoformat(str(start)[:10])
            lines.append(f"Дата консультации: {consult_date.isoformat()}")
        except ValueError:
            pass

    if patient.get("birthDate"):
        lines.append(f"Дата рождения: {str(patient['birthDate'])[:10]}")
    if patient.get("gender"):
        sex_ru = "мужской" if str(patient["gender"]).lower() == "male" else "женский"
        lines.append(f"Пол: {sex_ru}")

    names = patient.get("name") or []
    if names and isinstance(names[0], dict):
        family = str(names[0].get("family") or "")
        gv = names[0].get("given") or []
        given = " ".join(str(x) for x in gv if x)
        fn = " ".join(x for x in (family, given) if x).strip()
        if fn:
            lines.append(f"Пациент: {fn}")

    vit = vitals or {}
    for obs in by_type.get("Observation") or []:
        code = _obs_code(obs)
        val = _observation_value(obs)
        if val and code not in vit:
            lines.append(f"Показатель ({code}): {val}")
    if vit:
        lines.append("Объективный статус: " + "; ".join(f"{k} {v}" for k, v in vit.items()))

    diag_lines = list(doc_diagnoses or [])
    for cond in by_type.get("Condition") or []:
        icd, text = _icd_from_condition(cond)
        line = f"{icd + ' ' if icd else ''}{text}".strip()
        if line and line not in diag_lines:
            diag_lines.append(line)
    for dl in diag_lines:
        lines.append(f"Диагноз: {dl}")

    for comp in by_type.get("Composition") or []:
        for sec in comp.get("section") or []:
            if not isinstance(sec, dict):
                continue
            title = str(sec.get("title") or "").strip()
            div = (sec.get("text") or {}).get("div") if isinstance(sec.get("text"), dict) else None
            if isinstance(div, str) and div.strip():
                plain = re.sub(r"<[^>]+>", " ", div)
                plain = re.sub(r"\s+", " ", plain).strip()
                if plain:
                    lines.append(f"{title + ': ' if title else ''}{plain}")

    return "\n".join(lines).strip()


def bundle_to_consultation_text(bundle: dict[str, Any]) -> str:
    """Синтетический текст КЗ из Bundle – для пайплайна и L0-скрининга."""
    return _synthesize_bundle_text(bundle)
