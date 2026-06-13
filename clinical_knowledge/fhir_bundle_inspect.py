"""Разбор FHIR BY Bundle для проверки готовности к ЦИСЗ.

Программа испытаний МИС v.1.3-4 (содержимое) + Протокол МИС ОЗ-ЦИСЗ v.1.4 (Composition/пакет).
"""
from __future__ import annotations

import re
from typing import Any

VITAL_CODES = frozenset({
    "arterial-blood-pressure",
    "heart-rate",
    "body-temperature",
    "respiratory-rate",
    "oxygen-saturation",
    "body-temperature",
    "pulse-rate",
})
ANTHRO_CODES = frozenset({"body-height", "body-weight", "body-mass-index"})
OBJECTIVE_PROFILE_MARKERS = ("ObservationObjective", "objective", "Objective")
SUBJECTIVE_PROFILE_MARKERS = ("ObservationSubjective", "subjective", "Subjective")
CONSULT_SERVICE_MARKERS = ("ServiceRequestConsult", "consult", "Consult")
FINAL_DIAGNOSIS_MARKERS = ("FinalDiagnosis", "KindOfDiagnosis")
PRELIMINARY_MARKERS = ("preliminary", "предварительн", "?", "suspected", "Suspected")


def resources_by_type(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
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


def _profile_text(res: dict[str, Any]) -> str:
    meta = res.get("meta") or {}
    profiles = meta.get("profile") or []
    return " ".join(str(p) for p in profiles).lower()


def _obs_codes(obs: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    cod = obs.get("code") or {}
    for c in cod.get("coding") or []:
        if isinstance(c, dict) and c.get("code"):
            codes.append(str(c["code"]).lower())
    return codes


def _has_vital_observation(observations: list[dict[str, Any]]) -> bool:
    for obs in observations:
        prof = _profile_text(obs)
        if "vitalsigns" in prof or "vital" in prof:
            return True
        if set(_obs_codes(obs)) & VITAL_CODES:
            return True
        if obs.get("component"):
            return True
    return False


def _has_anthro_observation(observations: list[dict[str, Any]]) -> bool:
    for obs in observations:
        prof = _profile_text(obs)
        if "anthropometric" in prof:
            return True
        if set(_obs_codes(obs)) & ANTHRO_CODES:
            return True
    return False


def _has_subjective_observation(observations: list[dict[str, Any]]) -> bool:
    for obs in observations:
        prof = _profile_text(obs)
        if any(m.lower() in prof for m in SUBJECTIVE_PROFILE_MARKERS):
            return True
        if obs.get("valueString") or (obs.get("valueCodeableConcept") or {}).get("text"):
            cat = obs.get("category") or []
            for c in cat:
                for coding in (c.get("coding") or [] if isinstance(c, dict) else []):
                    if str(coding.get("code") or "").lower() in ("exam", "survey"):
                        if obs.get("valueString"):
                            return True
    for obs in observations:
        if obs.get("valueString") and len(str(obs.get("valueString"))) > 10:
            prof = _profile_text(obs)
            if "vital" not in prof and "anthropo" not in prof:
                return True
    return False


def _has_objective_observation(observations: list[dict[str, Any]]) -> bool:
    for obs in observations:
        prof = _profile_text(obs)
        if any(m.lower() in prof for m in OBJECTIVE_PROFILE_MARKERS):
            return True
        # Многосистемный осмотр: несколько component или valueString с системами
        comps = obs.get("component") or []
        if len(comps) >= 2:
            return True
    return False


def _condition_icd(cond: dict[str, Any]) -> str | None:
    code_obj = cond.get("code") or {}
    for c in code_obj.get("coding") or []:
        if not isinstance(c, dict):
            continue
        sys = str(c.get("system") or "")
        if "InternClassificDiseases" in sys or "icd" in sys.lower():
            raw = str(c.get("code") or "").strip().upper()
            return raw or None
    return None


def _is_final_diagnosis(cond: dict[str, Any]) -> bool:
    prof = _profile_text(cond)
    if "finaldiagnosis" in prof.replace("-", "").replace("_", ""):
        return True
    for ext in cond.get("extension") or []:
        if not isinstance(ext, dict):
            continue
        url = str(ext.get("url") or "")
        if "KindOfDiagnosis" in url:
            vcc = ext.get("valueCodeableConcept") or {}
            for coding in vcc.get("coding") or []:
                code = str(coding.get("code") or "").lower()
                display = str(coding.get("display") or "").lower()
                if code in ("final", "заключительный") or "заключительн" in display:
                    return True
    text_blob = str((cond.get("code") or {}).get("text") or "").lower()
    if any(m in text_blob for m in PRELIMINARY_MARKERS):
        return False
    if "?" in text_blob:
        return False
    return bool(_condition_icd(cond))


def _condition_has_clinical_status(cond: dict[str, Any]) -> bool:
    cs = cond.get("clinicalStatus") or {}
    return bool(cs.get("coding"))


def _condition_has_author(cond: dict[str, Any]) -> bool:
    return bool(cond.get("participant") or cond.get("recorder"))


def _encounter_status_completed(enc: dict[str, Any]) -> bool:
    return str(enc.get("status") or "").lower() == "completed"


def _encounter_has_period(enc: dict[str, Any]) -> bool:
    period = enc.get("actualPeriod") or enc.get("period") or {}
    return bool(period.get("start"))


def _encounter_has_participant(enc: dict[str, Any]) -> bool:
    return bool(enc.get("participant"))


def _encounter_diagnosis_link(enc: dict[str, Any]) -> bool:
    for d in enc.get("diagnosis") or []:
        if isinstance(d, dict) and d.get("condition"):
            return True
    return False


def _encounter_referral_link(enc: dict[str, Any]) -> bool:
    if enc.get("basedOn"):
        return True
    for item in enc.get("reasonReference") or []:
        if isinstance(item, dict) and item.get("reference"):
            return True
    return False


def _is_consult_service_request(sr: dict[str, Any]) -> bool:
    prof = _profile_text(sr)
    if any(m.lower() in prof for m in CONSULT_SERVICE_MARKERS):
        return True
    for c in (sr.get("code") or {}).get("coding") or []:
        disp = str(c.get("display") or "").lower()
        if "консультац" in disp or "прием" in disp or "приём" in disp:
            return True
    return str(sr.get("intent") or "").lower() in ("directive", "order", "plan")


def _medication_linked_to_encounter(mr: dict[str, Any]) -> bool:
    if mr.get("encounter"):
        return True
    for ext in mr.get("extension") or []:
        if isinstance(ext, dict) and ext.get("valueReference"):
            return True
    return False


def _first_entry_resource(bundle: dict[str, Any]) -> dict[str, Any] | None:
    entries = bundle.get("entry") or []
    if not entries or not isinstance(entries[0], dict):
        return None
    res = entries[0].get("resource")
    return res if isinstance(res, dict) else None


def _bundle_has_package_profile(bundle: dict[str, Any]) -> bool:
    meta = bundle.get("meta") or {}
    prof = " ".join(str(p) for p in meta.get("profile") or []).lower()
    if "medicationdocument" in prof.replace("-", "").replace("_", ""):
        return True
    if "пакет" in prof or "packagedocument" in prof:
        return True
    # Допуск: type=document без profile - частичное соответствие (МИС в разработке)
    return str(bundle.get("type") or "").lower() == "document" and bool(prof)


def _composition_subject_ok(comp: dict[str, Any]) -> bool:
    subj = comp.get("subject")
    if isinstance(subj, list):
        return any(isinstance(s, dict) and s.get("reference") for s in subj)
    if isinstance(subj, dict):
        return bool(subj.get("reference"))
    return False


def _composition_author_ok(comp: dict[str, Any]) -> bool:
    for item in comp.get("author") or []:
        if isinstance(item, dict) and item.get("reference"):
            return True
    return False


def _composition_custodian_ok(comp: dict[str, Any]) -> bool:
    cust = comp.get("custodian")
    return isinstance(cust, dict) and bool(cust.get("reference"))


def _composition_event_has_refs(comp: dict[str, Any]) -> bool:
    for ev in comp.get("event") or []:
        if not isinstance(ev, dict):
            continue
        for det in ev.get("detail") or []:
            if not isinstance(det, dict):
                continue
            ref = det.get("reference")
            if isinstance(ref, dict) and ref.get("reference"):
                return True
            if isinstance(ref, str) and ref.strip():
                return True
    # Секции Composition как слабый признак связности narrative
    for sec in comp.get("section") or []:
        if isinstance(sec, dict) and (sec.get("entry") or sec.get("text")):
            return True
    return False


def inspect_protocol_v14_checks(bundle: dict[str, Any]) -> dict[str, bool]:
    """Проверки структуры пакета по Протоколу взаимодействия МИС ОЗ - ЦИСЗ v.1.4."""
    compositions = resources_by_type(bundle).get("Composition") or []
    comp = compositions[0] if compositions else {}
    first = _first_entry_resource(bundle)
    first_is_comp = bool(first and first.get("resourceType") == "Composition")
    type_obj = comp.get("type") or {}
    has_type = any(
        isinstance(c, dict) and c.get("code")
        for c in type_obj.get("coding") or []
    )
    bundle_id = bundle.get("identifier") or comp.get("identifier")
    has_bundle_id = bool(bundle_id)
    return {
        "bundle_type_document": str(bundle.get("type") or "").lower() == "document",
        "bundle_profile_package": _bundle_has_package_profile(bundle),
        "bundle_identifier": has_bundle_id,
        "bundle_timestamp": bool(bundle.get("timestamp")),
        "composition_first_entry": first_is_comp,
        "composition_present": bool(compositions),
        "composition_status": bool(comp.get("status")),
        "composition_type": has_type,
        "composition_subject": _composition_subject_ok(comp) if compositions else False,
        "composition_encounter": bool(
            isinstance(comp.get("encounter"), dict) and comp.get("encounter", {}).get("reference")
        )
        if compositions
        else False,
        "composition_author": _composition_author_ok(comp) if compositions else False,
        "composition_custodian": _composition_custodian_ok(comp) if compositions else False,
        "composition_date": bool(comp.get("date")),
        "composition_event_links": _composition_event_has_refs(comp) if compositions else False,
    }


def detect_bundle_scenario(bundle: dict[str, Any]) -> str:
    """auto: specialist_consult если есть ServiceRequest консультации, иначе primary_ambulatory."""
    by_type = resources_by_type(bundle)
    for sr in by_type.get("ServiceRequest") or []:
        if _is_consult_service_request(sr):
            return "specialist_consult"
    for enc in by_type.get("Encounter") or []:
        if _encounter_referral_link(enc):
            return "specialist_consult"
    return "primary_ambulatory"


def inspect_bundle_checks(bundle: dict[str, Any]) -> dict[str, bool]:
    """Все булевы проверки по bundle."""
    by_type = resources_by_type(bundle)
    patients = by_type.get("Patient") or []
    encounters = by_type.get("Encounter") or []
    conditions = by_type.get("Condition") or []
    observations = by_type.get("Observation") or []
    service_requests = by_type.get("ServiceRequest") or []
    medications = by_type.get("MedicationRequest") or []
    lists_ = by_type.get("List") or []

    has_consult_sr = any(_is_consult_service_request(sr) for sr in service_requests)
    has_meds = bool(medications) or any(
        "prescription" in _profile_text(lst).lower() for lst in lists_
    )

    enc = encounters[0] if encounters else {}
    cond = conditions[0] if conditions else {}

    links_ok = bool(patients) and bool(encounters)
    if patients and encounters:
        subj = encounters[0].get("subject") or {}
        if isinstance(subj, dict) and subj.get("reference"):
            links_ok = True
        if conditions:
            csub = conditions[0].get("subject") or {}
            links_ok = links_ok and isinstance(csub, dict) and bool(csub.get("reference"))

    protocol = inspect_protocol_v14_checks(bundle)
    clinical = {
        "patient": bool(patients),
        "encounter": bool(encounters),
        "encounter_completed": _encounter_status_completed(enc) if encounters else False,
        "encounter_period": _encounter_has_period(enc) if encounters else False,
        "encounter_participant": _encounter_has_participant(enc) if encounters else False,
        "encounter_diagnosis_link": _encounter_diagnosis_link(enc) if encounters else False,
        "encounter_referral_link": _encounter_referral_link(enc) if encounters else False,
        "service_request_consult": has_consult_sr,
        "complaints": _has_subjective_observation(observations),
        "vitals": _has_vital_observation(observations),
        "anthropometrics": _has_anthro_observation(observations),
        "objective_exam": _has_objective_observation(observations),
        "diagnosis_icd10": any(_condition_icd(c) for c in conditions),
        "diagnosis_final_kind": any(_is_final_diagnosis(c) for c in conditions) if conditions else False,
        "diagnosis_clinical_status": any(_condition_has_clinical_status(c) for c in conditions),
        "diagnosis_author": any(_condition_has_author(c) for c in conditions),
        "bundle_links": links_ok,
        "medication_request": has_meds,
        "medication_encounter_link": (
            any(_medication_linked_to_encounter(m) for m in medications) if medications else False
        ),
    }
    return {**protocol, **clinical}


def inspect_text_checks(text: str) -> dict[str, bool]:
    """Эвристики для PDF/текста КЗ (частичный чек-лист 3.2.1)."""
    t = text or ""
    tl = t.lower()
    has_icd = bool(re.search(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b", t, re.I))
    preliminary = bool(re.search(r"\?|предварительн|подозрени", tl))
    complaints = bool(re.search(r"жалоб", tl))
    objective = bool(re.search(r"объективн|status praesens|ст\.?\s*localis", tl))
    vitals = bool(
        re.search(r"ад\s+\d|артериальн|температур|чсс|пульс|мм\s*рт", tl)
    )
    anamnesis = bool(re.search(r"анамнез", tl))
    treatment = bool(re.search(r"рекомендац|лечени|назначен", tl))
    follow_up = bool(re.search(r"повторн|контроль|явк", tl))
    doctor = bool(re.search(r"врач|доктор", tl))
    date = bool(re.search(r"дата\s+консультац|дата\s+приём", tl))
    return {
        "patient": bool(re.search(r"пациент|ф\.?\s*и\.?\s*о|дата рождения|пол:", tl)),
        "encounter": date or bool(re.search(r"консультац|приём|прием", tl)),
        "encounter_completed": True,  # текст не знает FHIR status
        "encounter_period": date,
        "encounter_participant": doctor,
        "encounter_diagnosis_link": has_icd and bool(re.search(r"диагноз", tl)),
        "encounter_referral_link": bool(re.search(r"направлен", tl)),
        "service_request_consult": bool(re.search(r"направлен.*консультац", tl)),
        "complaints": complaints,
        "vitals": vitals,
        "anthropometrics": bool(re.search(r"рост|вес|имт|масса тела", tl)),
        "objective_exam": objective,
        "diagnosis_icd10": has_icd,
        "diagnosis_final_kind": has_icd and not preliminary,
        "diagnosis_clinical_status": has_icd,
        "diagnosis_author": doctor,
        "bundle_links": has_icd and (complaints or objective),
        "medication_request": treatment,
        "medication_encounter_link": treatment and date,
        "text_anamnesis": anamnesis,
        "text_follow_up": follow_up,
    }
