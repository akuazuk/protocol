"""Сигналы из текста МО/КЗ для правил клинической согласованности.

Детерминированный слой (без LLM): сторона, суставы, длительность, план, audience.
См. docs/plans/2026-08-05-mo-eval-smirnova-concordance-v1.md.
"""
from __future__ import annotations

import re
from typing import Any

_SIDE_RIGHT = re.compile(r"\b(прав(?:ый|ая|ое|ом|ую)|right)\b", re.I)
_SIDE_LEFT = re.compile(r"\b(лев(?:ый|ая|ое|ом|ую)|left)\b", re.I)

_JOINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("knee", re.compile(r"колен", re.I)),
    ("hip", re.compile(r"тазобедрен", re.I)),
    ("ankle", re.compile(r"голеностоп", re.I)),
    ("shoulder", re.compile(r"плечев", re.I)),
    ("elbow", re.compile(r"локтев", re.I)),
    ("wrist", re.compile(r"лучезапяст", re.I)),
)

_EDEMA = re.compile(r"\b(от[её]к\w*|выпот\w*|припухл\w*|сглаженност\w*|от[её]чен\w*)\b", re.I)
_TENDERNESS = re.compile(r"\b(болезненн\w*|боль при пальпац\w*)\b", re.I)
_LIMP = re.compile(r"\bхромот\w*\b", re.I)
_MUSCLE = re.compile(r"\b(мышц\w*|миозит\w*|rectus|прямо[йе] мышцы)\b", re.I)

_DURATION = re.compile(
    r"(?:болеет|в течение|на протяжени\w*|около)?\s*"
    r"(\d+)\s*(месяц(?:а|ев)?|мес\.?|недел\w*|нед\.?|дн(?:я|ей|\.)?|день)",
    re.I,
)

_IMAGING = re.compile(r"\b(узи|рентген\w*|мрт|кт|r[oо]ntgen|x-?ray|ультразвуков)\b", re.I)
_LABS = re.compile(r"\b(оак|соэ|срб|с\-реактивн|анализ крови|креатинкиназ|кфк|ck)\b", re.I)
_FOLLOW_WORSENING = re.compile(
    r"при\s+(отрицательн\w*\s+динамик\w*|ухудшен\w*|отрицательной динамике)",
    re.I,
)
_BILATERAL = re.compile(r"\b(обеих|обоих|двусторонн\w*|с обеих|both)\b", re.I)
_TRAUMA = re.compile(r"\bтравм\w*\b", re.I)
_FEVER = re.compile(r"\b(лихорадк\w*|температур\w+|фебрильн\w*)\b", re.I)
_DYNAMICS = re.compile(r"\b(динамик\w*|ухудш\w*|улучш\w*|прогресс\w*)\b", re.I)
_LOAD = re.compile(r"\b(нагрузк\w*|спорт\w*|трениров\w*|физическ\w+\s+активност\w*)\b", re.I)
_INFECTION = re.compile(
    r"\b(инфекц\w*|гной\w*|абсцесс\w*|флегмон\w*|гипертерм\w*|сепсис\w*|покраснен\w*|флуктуац\w*)\b",
    re.I,
)

_DX_COVER_JOINT = {
    "knee": re.compile(r"(колен|гонарт|m17|m23|m25\.4|m08|синовит|артрит)", re.I),
    "hip": re.compile(r"(тазобедрен|cox|m16|m91|пертес|легг)", re.I),
    "ankle": re.compile(r"(голеностоп|m19|артрит|синовит)", re.I),
    "shoulder": re.compile(r"(плеч|m75|артрит|синовит)", re.I),
    "elbow": re.compile(r"(локтев|артрит|синовит)", re.I),
    "wrist": re.compile(r"(лучезапяст|запяст|артрит|синовит)", re.I),
}

# «травматолог» в плане - не закрытие DDx травмы; только явная травма/код/суставной DDx.
_PED_LIMP_DX_OK = re.compile(
    r"(?:m08|m91|m25|m00|m02|s[89]\d|t[89]\d|артрит|синовит|пертес|легг|"
    r"перелом|вывих|травм(?!атолог))",
    re.I,
)


def _txt(case: dict[str, Any], *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        val = case.get(key)
        if val:
            parts.append(str(val).strip())
    return " ".join(parts).strip()


def _duration_days(text: str) -> int | None:
    match = _DURATION.search(text or "")
    if not match:
        return None
    n = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("мес"):
        return n * 30
    if unit.startswith("нед"):
        return n * 7
    return n


def _side(text: str) -> str | None:
    has_r = bool(_SIDE_RIGHT.search(text or ""))
    has_l = bool(_SIDE_LEFT.search(text or ""))
    if has_r and not has_l:
        return "right"
    if has_l and not has_r:
        return "left"
    if has_r and has_l:
        return "both"
    return None


def _audience(case: dict[str, Any]) -> str:
    age = case.get("patient_age_years")
    try:
        if age is not None and float(age) < 18:
            return "pediatric"
    except (TypeError, ValueError):
        pass
    return "adult" if age is not None else "unknown"


def extract_mo_case_signals(case: dict[str, Any]) -> dict[str, Any]:
    """Извлечь сигналы для concordance findings."""
    complaints = _txt(case, "complaints")
    anamnesis = _txt(case, "anamnesis_doctor", "anamnesis_auto")
    objective = _txt(case, "objective_status", "exam_data", "st_localis")
    diagnosis = _txt(case, "clinical_diagnosis", "diagnosis_main_text", "diagnosis_list")
    treatment = _txt(case, "treatment_recommendations")
    exams_plan = _txt(case, "exam_recommendations")
    plan = f"{treatment} {exams_plan}".strip()
    clinical_blob = f"{complaints} {anamnesis} {objective}".strip()
    full = f"{clinical_blob} {diagnosis} {plan}".strip()

    icd = str(case.get("mkb_code_main") or case.get("icd10") or "").strip().upper()
    if not icd:
        m = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", diagnosis)
        if m:
            icd = m.group(1).upper()

    joint_edema: list[dict[str, str]] = []
    for joint, pat in _JOINTS:
        for m in pat.finditer(objective or ""):
            start = max(0, m.start() - 80)
            end = min(len(objective), m.end() + 80)
            window = objective[start:end]
            if _EDEMA.search(window):
                joint_edema.append(
                    {
                        "joint": joint,
                        "side": _side(window) or _side(objective) or "unknown",
                        "evidence": window.strip()[:220],
                    }
                )
                break

    duration_days = _duration_days(f"{complaints} {anamnesis}")
    complaint_side = _side(complaints) or _side(clinical_blob)
    anamnesis_flags = {
        "trauma": bool(_TRAUMA.search(anamnesis)),
        "fever": bool(_FEVER.search(anamnesis)),
        "dynamics": bool(_DYNAMICS.search(anamnesis)),
        "load": bool(_LOAD.search(anamnesis)),
        "trauma_addressed": bool(_TRAUMA.search(anamnesis)),
    }
    history_themes = sum(
        1
        for key in ("trauma_addressed", "fever", "dynamics", "load")
        if anamnesis_flags.get(key)
    )

    return {
        "audience": _audience(case),
        "patient_age_years": case.get("patient_age_years"),
        "complaints": complaints,
        "anamnesis": anamnesis,
        "objective": objective,
        "diagnosis": diagnosis,
        "plan": plan,
        "icd": icd,
        "duration_days": duration_days,
        "has_limp": bool(_LIMP.search(clinical_blob)),
        "has_muscle_finding": bool(_MUSCLE.search(objective) and _TENDERNESS.search(objective)),
        "joint_edema": joint_edema,
        "complaint_side": complaint_side,
        "plan_bilateral": bool(_BILATERAL.search(plan)),
        "plan_has_imaging": bool(_IMAGING.search(plan)),
        "plan_has_labs": bool(_LABS.search(plan)),
        "follow_up_on_worsening_only": bool(_FOLLOW_WORSENING.search(plan)),
        "anamnesis_flags": anamnesis_flags,
        "anamnesis_theme_count": history_themes,
        "anamnesis_len": len(anamnesis),
        "infection_signs": bool(_INFECTION.search(full)),
        "dx_covers_joint": {
            joint: bool(_DX_COVER_JOINT[joint].search(f"{diagnosis} {icd}"))
            for joint, _ in _JOINTS
        },
        # Диагноз/МКБ - основной сигнал; план учитываем только если явно назван DDx
        # (не «осмотр травматолога» как follow-up).
        "ped_limp_dx_ok": bool(
            _PED_LIMP_DX_OK.search(f"{diagnosis} {icd}")
            or _PED_LIMP_DX_OK.search(plan or "")
        ),
    }
