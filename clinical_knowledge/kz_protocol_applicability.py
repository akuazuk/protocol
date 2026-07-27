"""Оценка применимости протокола к КЗ с разделением уверенностей (Workstream E).

Жёсткие инварианты (§9.2 ТЗ overnight-v1):
- детский КП не штрафует взрослое КЗ (и наоборот);
- стационарный КП не штрафует амбулаторное КЗ;
- реабилитационный КП не заменяет диагностический;
- общий организационный КП не считается disease-specific;
- protocol fallback не штрафует;
- при ``applicability_confidence < 0.75`` требования протокола advisory-only.

КЗ (консультативное заключение) по определению - амбулаторный документ, поэтому
care setting КЗ = ``outpatient``.
"""
from __future__ import annotations

from typing import Any

from .kz_evaluation_schema import ProtocolMatchInfo
from .rule_trust import TRUST_C, TRUST_D

APPLICABILITY_ADVISORY_CUTOFF = 0.75

_PED_MARKERS = ("детск", "педиатр", "ребён", "ребен", "новорожд", "неонат", "перинат")
_ADULT_MARKERS = ("взросл",)
_INPATIENT_MARKERS = ("стационар", "круглосуточн", "стац. услов", "стац услов")
_REHAB_MARKERS = ("реабилитац", "медицинская реабилитация", "восстановительн")
_GENERIC_MARKERS = (
    "организац", "порядок оказания", "общие требования", "маршрутизац населения",
    "профилактическ", "диспансеризац населения",
)


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _norm_match_score(raw: Any) -> float | None:
    """match_score может быть 0-1 или 0-100 -> к 0-1."""
    if raw is None:
        return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if f > 1.5:
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _patient_adult_or_child(case: dict[str, Any]) -> str:
    aud = str(case.get("adult_or_child") or "").strip().lower()
    if aud in ("adult", "child", "newborn"):
        return "child" if aud == "newborn" else aud
    age = case.get("patient_age_years")
    try:
        age = int(float(age)) if age not in (None, "") else None
    except (TypeError, ValueError):
        age = None
    if age is None:
        return "unknown"
    return "child" if age < 18 else "adult"


def _protocol_population(protocol_ctx: Any) -> str:
    pop = str(_get(protocol_ctx, "population") or "").strip().lower()
    if pop in ("adult", "child", "newborn", "pregnant", "any", "adult_and_child"):
        return "child" if pop == "newborn" else pop
    name = str(_get(protocol_ctx, "name") or "").lower()
    has_ped = any(m in name for m in _PED_MARKERS)
    has_adult = any(m in name for m in _ADULT_MARKERS)
    if has_ped and not has_adult:
        return "child"
    if has_adult and not has_ped:
        return "adult"
    return "any"


def _protocol_care_setting(protocol_ctx: Any) -> str:
    cs = str(_get(protocol_ctx, "care_setting") or "").strip().lower()
    if cs in ("inpatient", "outpatient", "mixed", "emergency", "rehabilitation", "palliative"):
        return cs
    name = str(_get(protocol_ctx, "name") or "").lower()
    if any(m in name for m in _INPATIENT_MARKERS):
        return "inpatient"
    return "unknown"


def assess_applicability(case: dict[str, Any], protocol_ctx: Any) -> ProtocolMatchInfo | None:
    """Вернуть ``ProtocolMatchInfo`` c уверенностями и penalty_eligible.

    ``None`` - если протокол не подобран (нечего оценивать).
    """
    if protocol_ctx is None:
        return None

    reasons: list[str] = []
    name = str(_get(protocol_ctx, "name") or "")
    lname = name.lower()

    retrieval = _norm_match_score(_get(protocol_ctx, "match_score"))
    if retrieval is None:
        # нет явного скора: подобран по коду МКБ -> умеренная уверенность
        retrieval = 0.7 if _get(protocol_ctx, "condition_id") else 0.4

    is_fallback = bool(_get(protocol_ctx, "is_fallback"))
    is_rehab = bool(_get(protocol_ctx, "is_rehabilitation")) or any(m in lname for m in _REHAB_MARKERS)
    is_generic = bool(_get(protocol_ctx, "is_generic")) or any(m in lname for m in _GENERIC_MARKERS)

    # population
    patient_pop = _patient_adult_or_child(case)
    proto_pop = _protocol_population(protocol_ctx)
    if proto_pop in ("any", "adult_and_child", "unknown", "") or patient_pop == "unknown":
        population_match: bool | None = True if patient_pop != "unknown" else None
    else:
        population_match = proto_pop == patient_pop
        if not population_match:
            reasons.append(f"population mismatch: protocol={proto_pop}, patient={patient_pop}")

    # care setting: КЗ амбулаторное
    proto_cs = _protocol_care_setting(protocol_ctx)
    if proto_cs == "inpatient":
        care_setting_match: bool | None = False
        reasons.append("protocol is inpatient, KZ is outpatient")
    elif proto_cs in ("outpatient", "mixed", "emergency", "unknown", ""):
        care_setting_match = True
    else:
        care_setting_match = True

    version_current = _get(protocol_ctx, "version_current")
    if version_current is None:
        version_current = None  # неизвестно
    else:
        version_current = bool(version_current)

    # applicability_confidence: старт от retrieval, домножаем на штрафы инвариантов
    conf = retrieval
    if population_match is False:
        conf *= 0.25
    if care_setting_match is False:
        conf *= 0.35
    if is_rehab:
        conf *= 0.3
        reasons.append("rehabilitation protocol not a diagnostic substitute")
    if is_generic:
        conf *= 0.3
        reasons.append("generic/organizational protocol is not disease-specific")
    if is_fallback:
        conf = min(conf, 0.3)
        reasons.append("fallback protocol -> advisory only")
    if version_current is False:
        conf *= 0.6
        reasons.append("protocol version not current")

    conf = round(max(0.0, min(1.0, conf)), 3)

    penalty_eligible = (
        conf >= APPLICABILITY_ADVISORY_CUTOFF
        and population_match is not False
        and care_setting_match is not False
        and not is_fallback
        and not is_generic
        and not is_rehab
    )
    if not penalty_eligible and not reasons:
        reasons.append(f"applicability_confidence {conf} < {APPLICABILITY_ADVISORY_CUTOFF}")

    trust_level = TRUST_C if penalty_eligible else TRUST_D

    return ProtocolMatchInfo(
        condition_id=_get(protocol_ctx, "condition_id"),
        name=name or None,
        applicability_confidence=conf,
        retrieval_confidence=retrieval,
        population_match=population_match,
        care_setting_match=care_setting_match,
        specialty_match=None,
        version_current=version_current,
        penalty_eligible=penalty_eligible,
        trust_level=trust_level,
        reasons=reasons,
    )
