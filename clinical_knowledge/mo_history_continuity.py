"""Непрерывность эпизода и отбор МО на глубокий прогон истории.

План: docs/plans/2026-08-14-mo-history-continuity-deep-run-v1.md

Слой A: дешёвый вердикт из бандла (коды / короткий текст).
Официальный overall_pct не меняем.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

ENGINE = "mo_history_continuity_v1"

MODE_KNOWN_DOCTOR = "known_episode_same_doctor"
MODE_KNOWN_SPECIALTY = "known_episode_same_specialty"
MODE_NEW_DOCTOR = "new_problem_known_doctor"
MODE_NEW_SPECIALTY = "new_problem_known_specialty"
MODE_OTHER = "other_history_only"
MODE_NONE = "no_history"

TRACK_SAFETY = "safety"
TRACK_HISTORY = "history"
TRACK_STRONG = "strong_model"
TRACK_SKIP = "skip"

POOR_BANDS = frozenset({"bad", "weak"})
_ICD_STEM_RE = re.compile(r"\b([A-TV-ZА-Яа-я]\d{2})", re.IGNORECASE)

MODE_LABEL_RU = {
    MODE_KNOWN_DOCTOR: "Продолжение случая у этого врача",
    MODE_KNOWN_SPECIALTY: "Тот же эпизод у коллег специальности",
    MODE_NEW_DOCTOR: "Новый повод у знакомого врача",
    MODE_NEW_SPECIALTY: "Новый повод, пациент уже был у специальности",
    MODE_OTHER: "Есть визиты к другим специальностям",
    MODE_NONE: "На складе нет более ранних визитов с ключом",
}

TRACK_LABEL_RU = {
    TRACK_SAFETY: "Сначала риск, потом история",
    TRACK_HISTORY: "Глубокий прогон с полной историей эпизода",
    TRACK_STRONG: "Сильная модель (история мало что объяснит)",
    TRACK_SKIP: "Глубокий прогон не нужен",
}


def _stem(code: str) -> str:
    text = (code or "").strip().upper().replace(",", ".")
    if len(text) >= 3 and text[0].isalpha() and text[1:3].isdigit():
        return text[:3]
    found = _ICD_STEM_RE.search(text)
    return found.group(1).upper() if found else ""


def _visit_text(visit: Mapping[str, Any]) -> str:
    for key in ("diagnosis_text", "diagnosis_short", "clinical_diagnosis", "text"):
        val = str(visit.get(key) or "").strip()
        if val:
            return val
    return ""


def _same_episode(
    *,
    current_stem: str,
    current_text: str,
    visit: Mapping[str, Any],
) -> bool:
    visit_stem = _stem(str(visit.get("diagnosis_code") or ""))
    if current_stem and visit_stem and current_stem == visit_stem:
        return True
    visit_text = _visit_text(visit)
    if current_text and visit_text:
        try:
            from clinical_knowledge.dx_query_expand import diagnosis_tokens

            left = set(diagnosis_tokens(current_text, min_len=4, limit=16))
            right = set(diagnosis_tokens(visit_text, min_len=4, limit=16))
            if left and right and len(left & right) / max(len(left), len(right)) >= 0.45:
                return True
        except Exception:  # noqa: BLE001
            cur = current_text.lower()
            prior = visit_text.lower()
            if len(cur) >= 6 and cur[:12] in prior:
                return True
    return False


def _poor_zones(row: Mapping[str, Any] | None) -> dict[str, bool]:
    rec = row if isinstance(row, Mapping) else {}
    dx = str(rec.get("zone2a_band") or "").lower() in POOR_BANDS
    plan = str(rec.get("zone2b_band") or "").lower() in POOR_BANDS
    form = str(rec.get("zone1_band") or "").lower() in POOR_BANDS
    return {"diagnosis": dx, "plan": plan, "documentation": form, "any_clinical": dx or plan}


def evaluate_history_continuity(
    *,
    current_code: str = "",
    current_text: str = "",
    history_bundle: Mapping[str, Any] | None = None,
    zones: Mapping[str, Any] | None = None,
    attention_primary: str = "",
    overall_pct: float | None = None,
    history_prior_n: int | None = None,
    history_tier: str = "",
) -> dict[str, Any]:
    """Вердикт непрерывности + приоритет глубокого прогона. Без patient_id."""
    bundle = history_bundle if isinstance(history_bundle, Mapping) else {}
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), Mapping) else {}
    same_doctor = [row for row in (bundle.get("same_doctor") or []) if isinstance(row, Mapping)]
    same_specialty = [
        row for row in (bundle.get("same_specialty") or []) if isinstance(row, Mapping)
    ]
    other = [row for row in (bundle.get("other") or []) if isinstance(row, Mapping)]
    stem = _stem(current_code or str(summary.get("current_code") or ""))
    text = (current_text or "").strip()

    matched_doctor = [
        row for row in same_doctor if _same_episode(current_stem=stem, current_text=text, visit=row)
    ]
    matched_specialty = [
        row
        for row in same_specialty
        if _same_episode(current_stem=stem, current_text=text, visit=row)
    ]

    if matched_doctor:
        mode = MODE_KNOWN_DOCTOR
        matched = matched_doctor
        shelf = "same_doctor"
    elif matched_specialty:
        mode = MODE_KNOWN_SPECIALTY
        matched = matched_specialty
        shelf = "same_specialty"
    elif same_doctor:
        mode = MODE_NEW_DOCTOR
        matched = []
        shelf = "same_doctor"
    elif same_specialty:
        mode = MODE_NEW_SPECIALTY
        matched = []
        shelf = "same_specialty"
    elif other:
        mode = MODE_OTHER
        matched = []
        shelf = "other"
    else:
        # Очередь дня часто без полного бандла - опираемся на warehouse tier.
        tier = str(history_tier or bundle.get("tier") or "")
        prior_n = int(summary.get("n_visits") or history_prior_n or 0)
        if tier == "known_to_doctor":
            mode = MODE_KNOWN_DOCTOR
            matched = []
            shelf = "same_doctor"
        elif tier == "known_in_specialty_only":
            mode = MODE_KNOWN_SPECIALTY
            matched = []
            shelf = "same_specialty"
        elif tier == "new_for_profile":
            mode = MODE_NEW_DOCTOR
            matched = []
            shelf = "same_doctor"
        elif other or prior_n > 0:
            mode = MODE_OTHER
            matched = []
            shelf = "other"
        else:
            mode = MODE_NONE
            matched = []
            shelf = ""

    already: list[str] = []
    if matched:
        already.append("diagnosis")
        if any(_visit_text(row) for row in matched):
            already.append("diagnosis_text")

    poor = _poor_zones(zones)
    attention = str(attention_primary or "").strip().lower()
    known_episode = mode in {MODE_KNOWN_DOCTOR, MODE_KNOWN_SPECIALTY}
    has_prior = mode not in {MODE_NONE}

    if attention == "safety":
        track = TRACK_SAFETY
        score = 200
    elif poor["any_clinical"] and known_episode:
        track = TRACK_HISTORY
        score = 150
    elif poor["any_clinical"] and has_prior:
        track = TRACK_HISTORY
        score = 80
    elif poor["any_clinical"]:
        track = TRACK_STRONG
        score = 40
    elif poor["documentation"]:
        track = TRACK_SKIP
        score = 10
    else:
        track = TRACK_SKIP
        score = 0

    last = matched[-1] if matched else {}
    return {
        "engine": ENGINE,
        "mode": mode,
        "mode_ru": MODE_LABEL_RU[mode],
        "shelf": shelf,
        "known_episode": known_episode,
        "already_described": already,
        "matched_n": len(matched),
        "prior_same_doctor_n": len(same_doctor),
        "prior_same_specialty_n": len(same_specialty),
        "last_matched_date": str(last.get("visit_date") or ""),
        "last_matched_code": str(last.get("diagnosis_code") or ""),
        "poor_diagnosis": poor["diagnosis"],
        "poor_plan": poor["plan"],
        "deep_run_track": track,
        "deep_run_track_ru": TRACK_LABEL_RU[track],
        "deep_run_score": score,
        "history_tier": str(history_tier or bundle.get("tier") or ""),
        "usage_ru": (
            "Официальную оценку визита не меняем. "
            "Если диагноз или план слабые, а эпизод уже вёлся - этот случай "
            "раньше идёт на глубокий прогон истории. "
            "Сегодняшний осмотр и актуальный план всё равно обязательны."
        ),
        "overall_pct": overall_pct,
    }


def rank_for_deep_run(item: Mapping[str, Any]) -> tuple[int, float, str]:
    """Ключ сортировки: выше deep_run_score, ниже формула, стабильный id."""
    score = int(item.get("deep_run_score") or 0)
    pct = item.get("overall_pct")
    try:
        pct_f = float(pct) if pct is not None and pct != "" else 999.0
    except (TypeError, ValueError):
        pct_f = 999.0
    return (-score, pct_f, str(item.get("case_id") or item.get("mis_id") or ""))


def attach_continuity_to_public_bundle(
    bundle: Mapping[str, Any] | None,
    continuity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = dict(bundle) if isinstance(bundle, Mapping) else {}
    if isinstance(continuity, Mapping):
        out["continuity"] = dict(continuity)
    return out
