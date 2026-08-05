"""Concordance findings для МО/КЗ (shadow-слой, отдельно от L1 overall).

См. docs/plans/2026-08-05-mo-eval-smirnova-concordance-v1.md.
"""
from __future__ import annotations

import os
import re
from typing import Any

from .mo_case_signals import extract_mo_case_signals

ENGINE = "mo_concordance_v1"
_SOURCE = "mo_concordance_v1"


def concordance_findings_enabled() -> bool:
    """Shadow findings. Default on; set MO_CONCORDANCE_FINDINGS=0 to disable."""
    raw = (os.environ.get("MO_CONCORDANCE_FINDINGS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def concordance_primary_enabled() -> bool:
    """Merge into primary deep findings (affects overall). Default off."""
    raw = (os.environ.get("MO_CONCORDANCE_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _finding(
    code: str,
    axis: str,
    severity: str,
    *,
    title: str,
    detail: str = "",
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "axis": axis,
        "severity": severity,
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "engine": ENGINE,
    }


def evaluate_mo_concordance(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Вернуть список непройденных concordance findings (shadow shape = deep finding)."""
    sig = extract_mo_case_signals(case)
    out: list[dict[str, Any]] = []

    # 1) finding_not_in_diagnosis
    for item in sig.get("joint_edema") or []:
        joint = str(item.get("joint") or "")
        covers = (sig.get("dx_covers_joint") or {}).get(joint)
        if joint and not covers:
            out.append(
                _finding(
                    "finding_not_in_diagnosis",
                    "clinical_concordance",
                    "P1",
                    title="Находка в статусе не отражена в диагнозе",
                    detail=f"Отёк/припухлость сустава ({joint}) есть в осмотре, но не покрыта диагнозом/МКБ",
                    evidence=str(item.get("evidence") or ""),
                )
            )

    # 2) anamnesis_thin_for_duration
    duration = sig.get("duration_days")
    if isinstance(duration, int) and duration >= 28:
        themes = int(sig.get("anamnesis_theme_count") or 0)
        anam_len = int(sig.get("anamnesis_len") or 0)
        if themes < 2 or anam_len < 80:
            out.append(
                _finding(
                    "anamnesis_thin_for_duration",
                    "documentation",
                    "P2",
                    title="Анамнез слишком краток для длительности жалобы",
                    detail=(
                        f"Длительность ~{duration} дн., тем анамнеза {themes}/4, "
                        f"длина текста {anam_len} символов"
                    ),
                    evidence=(sig.get("anamnesis") or "")[:300],
                )
            )

    # 3) underworkup_chronic_red_flag
    pediatric = sig.get("audience") == "pediatric"
    chronic = isinstance(duration, int) and duration >= 28
    red_presentation = bool(sig.get("has_limp") or sig.get("joint_edema"))
    no_workup = not (sig.get("plan_has_imaging") or sig.get("plan_has_labs"))
    worsening_only = bool(sig.get("follow_up_on_worsening_only"))
    if chronic and red_presentation and no_workup and worsening_only:
        sev = "P1" if pediatric else "P2"
        out.append(
            _finding(
                "underworkup_chronic_red_flag",
                "safety",
                sev,
                title="Недостаточный объём обследования при хроническом сценарии",
                detail=(
                    "Длительные жалобы с хромотой/отёком сустава при отсутствии "
                    "imaging/labs в плане и контроле только при ухудшении"
                ),
                evidence=(sig.get("plan") or "")[:300],
            )
        )

    # 4) plan_laterality_mismatch
    side = sig.get("complaint_side")
    if side in {"right", "left"} and sig.get("plan_bilateral"):
        out.append(
            _finding(
                "plan_laterality_mismatch",
                "clinical_concordance",
                "P3",
                title="Латеральность плана не совпадает с жалобой",
                detail="Жалоба/статус односторонние, в плане двусторонняя процедура без обоснования",
                evidence=f"side={side}; plan={(sig.get('plan') or '')[:220]}",
            )
        )

    # 5) icd_weakly_supported (start: M60*)
    icd = str(sig.get("icd") or "")
    if re.match(r"^M60", icd) and not sig.get("infection_signs"):
        out.append(
            _finding(
                "icd_weakly_supported",
                "clinical_concordance",
                "P2",
                title="Код МКБ слабо поддержан клинической картиной",
                detail="M60* при отсутствии признаков инфекции/системного воспаления в тексте",
                evidence=f"ICD={icd}; dx={(sig.get('diagnosis') or '')[:200]}",
            )
        )

    # 6) pediatric_limp_ddx_not_addressed
    if (
        pediatric
        and sig.get("has_limp")
        and isinstance(duration, int)
        and duration >= 28
        and not sig.get("ped_limp_dx_ok")
    ):
        out.append(
            _finding(
                "pediatric_limp_ddx_not_addressed",
                "clinical_concordance",
                "P2",
                title="Не закрыт детский DDx длительной хромоты",
                detail="Нет диагноза/плана из ожидаемого ряда (ЮА, Пертес, травма, инфекция сустава и т.п.)",
                evidence=f"dx={(sig.get('diagnosis') or '')[:200]}; icd={icd}",
            )
        )

    return out
