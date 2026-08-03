"""Фильтрация протоколов по возрасту, специальности и МКБ (B2C post-filter)."""
from __future__ import annotations

import copy
from typing import Any

from .protocol_audience import infer_protocol_audience, norm_audience_blob

_PEDiatric_TITLE_NEEDLES = (
    "детск",
    "дет нас",
    "д-нас",
    "детс",
    "неонат",
    "новорожд",
    "pediatr",
    "детей",
)
_CHILD_PROTOCOL_HARD_REJECT = (
    "инсульт у дет",
    "у детей",
    "детское население",
    "smn1",
    "smn2",
    "pmp22",
)
_NEUROLOGY_NEEDLES = ("невролог", "nevrolog", "головн", "позвоноч", "радикул", "мигрен", "ишиас")
_SPINE_ICD = frozenset({"M51", "M53", "M54"})


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _protocol_blob(path: str, title: str = "") -> str:
    return norm_audience_blob(f"{path} {title}")


def should_reject_protocol(
    *,
    path: str,
    title: str,
    patient_context: dict[str, Any],
) -> tuple[bool, str]:
    """True если протокол не подходит пациенту."""
    blob = _protocol_blob(path, title)
    age_group = (patient_context.get("age_group") or "").lower()
    specialty = (patient_context.get("specialty") or "").lower()
    icd_list = [str(c).upper() for c in (patient_context.get("icd10_codes") or [])]

    aud = infer_protocol_audience(path, title)
    if age_group == "adult" and aud == "pediatric":
        return True, "pediatric_for_adult"
    if age_group == "child" and aud == "adult":
        return True, "adult_for_child"

    for needle in _CHILD_PROTOCOL_HARD_REJECT:
        if needle in blob:
            if age_group != "child":
                return True, f"hard_reject:{needle}"

    if specialty == "neurology" and icd_list:
        spine = any(_icd_root(c) in _SPINE_ICD or c.startswith("M5") for c in icd_list)
        if spine:
            has_neuro = any(n in blob for n in _NEUROLOGY_NEEDLES)
            has_ped = any(n in blob for n in _PEDiatric_TITLE_NEEDLES)
            bladder = any(n in blob for n in ("мочев", "пузыр", "уролог", "цистит"))
            if has_ped and not has_neuro:
                return True, "pediatric_not_neurology"
            if bladder and not has_neuro:
                return True, "bladder_not_spine"

    return False, ""


def _filter_match_list(
    matches: list[Any],
    patient_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    reasons: list[str] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        path = str(m.get("source_path") or m.get("local_path") or "")
        title = str(m.get("title") or "")
        reject, reason = should_reject_protocol(path=path, title=title, patient_context=patient_context)
        if reject:
            reasons.append(reason)
            continue
        kept.append(m)
    return kept, reasons


def filter_l1_protocols(
    l1_result: dict[str, Any],
    patient_context: dict[str, Any],
) -> dict[str, Any]:
    """Post-filter matches, alignment cards и пересчёт primary protocol."""
    out = copy.deepcopy(l1_result)
    sa = out.get("structured_analysis")
    if not isinstance(sa, dict):
        sa = {}
        out["structured_analysis"] = sa

    matches = list(sa.get("matches") or [])
    filtered_matches, match_reasons = _filter_match_list(matches, patient_context)
    sa["matches"] = filtered_matches
    out["matched_protocols_count"] = len(filtered_matches)

    align = out.get("alignment")
    if isinstance(align, dict):
        cards = list(align.get("alignment_cards") or [])
        new_cards: list[dict[str, Any]] = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            path = str(card.get("protocol_path") or "")
            title = str(card.get("protocol_title") or card.get("name_ru") or "")
            reject, _ = should_reject_protocol(path=path, title=title, patient_context=patient_context)
            if reject:
                continue
            new_cards.append(card)
        align["alignment_cards"] = new_cards

        audit = align.get("audit_trail")
        if isinstance(audit, dict):
            pm = list(audit.get("protocol_matches") or [])
            fm, _ = _filter_match_list(pm, patient_context)
            audit["protocol_matches"] = fm
            paths = [str(m.get("source_path") or m.get("local_path") or "") for m in fm if isinstance(m, dict)]
            audit["protocol_paths"] = [p for p in paths if p]

    out["_protocol_filter"] = {
        "removed_count": max(0, len(matches) - len(filtered_matches)),
        "reasons": match_reasons[:6],
    }
    return out


def compute_protocol_match_confidence(
    patient_context: dict[str, Any],
    l1_result: dict[str, Any],
) -> tuple[float, str]:
    """0.0-1.0 и bucket low|medium|high."""
    sa = l1_result.get("structured_analysis") if isinstance(l1_result.get("structured_analysis"), dict) else {}
    matches = sa.get("matches") if isinstance(sa.get("matches"), list) else []
    filt = l1_result.get("_protocol_filter") if isinstance(l1_result.get("_protocol_filter"), dict) else {}
    removed = int(filt.get("removed_count") or 0)

    score = 0.15
    age_group = patient_context.get("age_group")
    if age_group in ("adult", "child"):
        score += 0.15
    if patient_context.get("icd10_codes"):
        score += 0.15
    if patient_context.get("specialty"):
        score += 0.10
    if matches:
        top = matches[0] if isinstance(matches[0], dict) else {}
        ms = top.get("match_score")
        if isinstance(ms, (int, float)):
            score += min(0.35, float(ms) / 100.0 * 0.35)
        else:
            score += 0.20
    if removed:
        score -= min(0.25, removed * 0.08)

    score = max(0.0, min(1.0, round(score, 2)))
    if score < 0.5:
        bucket = "low"
    elif score < 0.75:
        bucket = "medium"
    else:
        bucket = "high"
    return score, bucket
