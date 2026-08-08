"""Оценка диагноза МО против справочника МКБ (отдельно от подбора КП).

См. docs/plans/2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md.
Пороги text_rubric_fit: ok ≥0.35, review ≥0.25 (как consult enrichment).
Default: shadow findings; primary только при MO_ICD_DIR_IN_PRIMARY=1.
"""
from __future__ import annotations

import os
import re
from typing import Any

from clinical_knowledge.mo_icd_thresholds import (  # noqa: E402
    DIR_HIT_SCORE_MIN as DIR_HIT_SCORE_MIN_DEFAULT,
    TEXT_FIT_OK as TEXT_FIT_OK_DEFAULT,
    TEXT_FIT_REVIEW as TEXT_FIT_REVIEW_DEFAULT,
    dir_hit_score_min as _dir_hit_score_min,
    pipeline_in_primary_enabled,
    text_fit_ok as _text_fit_ok,
    text_fit_review as _text_fit_review,
)

ENGINE = "mo_icd_directory_v1"
_SOURCE = "mo_icd_directory_v1"

# Совместимость тестов; runtime - getters.
TEXT_FIT_OK = TEXT_FIT_OK_DEFAULT
TEXT_FIT_REVIEW = TEXT_FIT_REVIEW_DEFAULT
DIR_HIT_SCORE_MIN = DIR_HIT_SCORE_MIN_DEFAULT


def icd_directory_eval_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_DIRECTORY_EVAL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def icd_directory_primary_enabled() -> bool:
    """Merge into primary findings (affects overall). Default off = shadow."""
    if pipeline_in_primary_enabled():
        return True
    raw = (os.environ.get("MO_ICD_DIR_IN_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _norm_tokens(text: str) -> set[str]:
    """Токены для coverage title; при MO_ICD_LIGHT_STEM - через общий light stem."""
    try:
        from clinical_knowledge.clinical_text_similarity import tokens as _shared_tokens

        return _shared_tokens(text, min_len=4)
    except Exception:  # noqa: BLE001
        return {t for t in re.findall(r"[а-яёa-z]{4,}", (text or "").lower()) if len(t) >= 4}


def title_match_score(diagnosis_text: str, ru_title: str | None) -> float:
    if not ru_title or not diagnosis_text:
        return 0.0
    dt = _norm_tokens(diagnosis_text)
    rt = _norm_tokens(ru_title)
    if not dt or not rt:
        return 0.0
    return len(dt & rt) / max(len(rt), 1)


def free_text_is_substantive(diagnosis_text: str) -> bool:
    """Есть ли осмысленная формулировка сверх токенов кода МКБ.

    Code-only / «F41.2» не считаем текстом для mismatch: валидный код в справочнике
    уже задаёт рубрику.
    """
    raw = (diagnosis_text or "").strip()
    if not raw:
        return False
    try:
        from clinical_knowledge.dx_query_expand import strip_icd_tokens

        cleaned = strip_icd_tokens(raw)
    except Exception:  # noqa: BLE001
        cleaned = raw
    return len(_norm_tokens(cleaned)) >= 2


def _finding(
    code: str,
    *,
    severity: str,
    title: str,
    detail: str = "",
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "axis": "icd_directory",
        "severity": severity,
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "engine": ENGINE,
        "linked_fields": ["clinical_diagnosis", "mis_diagnos", "mkb_code_main"],
        "link_hint_ru": "Сверьте формулировку диагноза со справочником МКБ",
    }


def evaluate_diagnosis_against_icd_directory(
    diag_text: str,
    codes: list[str] | None = None,
) -> dict[str, Any]:
    """Сверка текста диагноза (±кодов) со справочником МКБ RU.

    Returns dict with directory_hit, code_in_directory, text_rubric_fit,
    verdict, score_pct, findings, candidates.
    """
    import icd_mkb

    text = (diag_text or "").strip()
    code_list = [str(c).strip().upper() for c in (codes or []) if str(c).strip()]
    # unique preserve order
    seen: set[str] = set()
    uniq_codes: list[str] = []
    for c in code_list:
        if c not in seen:
            seen.add(c)
            uniq_codes.append(c)

    candidates: list[dict[str, Any]] = []
    directory_hit = False
    top_lex = 0.0
    if len(text) >= 3:
        try:
            suggestions = icd_mkb.suggest_icd_from_russian(text, max_results=5)
        except Exception:  # noqa: BLE001
            suggestions = []
        for row in suggestions:
            if not isinstance(row, dict):
                continue
            score = float(row.get("score") or 0)
            top_lex = max(top_lex, score)
            candidates.append(
                {
                    "code": row.get("code"),
                    "title_ru": row.get("title_ru"),
                    "score": score,
                    "match_method": row.get("match_method"),
                }
            )
        directory_hit = bool(candidates) and top_lex >= _dir_hit_score_min()

    thr_ok = _text_fit_ok()
    thr_review = _text_fit_review()

    code_checks: list[dict[str, Any]] = []
    text_rubric_fit = 0.0
    any_code_in_dir = False
    any_code_unknown = False
    for code in uniq_codes:
        in_dir = False
        title = None
        try:
            in_dir = bool(icd_mkb.is_code_in_ru_reference(code))
            title = icd_mkb.ru_title(code) if in_dir else None
        except Exception:  # noqa: BLE001
            in_dir = False
            title = None
        fit = title_match_score(text, title) if text and title else 0.0
        text_rubric_fit = max(text_rubric_fit, fit)
        if in_dir:
            any_code_in_dir = True
        else:
            any_code_unknown = True
        code_checks.append(
            {
                "code": code,
                "in_directory": in_dir,
                "title_ru": title,
                "text_rubric_fit": round(fit, 3),
            }
        )

    findings: list[dict[str, Any]] = []
    if not text and not uniq_codes:
        # Главная проверка владельца: диагноз должен быть в МО
        absent = _finding(
            "B_dx_absent",
            severity="P1",
            title="Диагноз отсутствует в МО",
            detail=(
                "Нет формулировки диагноза и кода МКБ в документе "
                "(слоты диагноза пусты, код не найден по полному тексту)."
            ),
            evidence="",
        )
        return {
            "engine": ENGINE,
            "directory_hit": False,
            "code_in_directory": None,
            "text_rubric_fit": 0.0,
            "verdict": "fail",
            "score_pct": 0,
            "findings": [absent],
            "candidates": [],
            "code_checks": [],
            "thresholds": {"text_fit_ok": thr_ok, "text_fit_review": thr_review},
        }

    if text and not directory_hit and not any_code_in_dir:
        findings.append(
            _finding(
                "B_icd_dir_no_match",
                severity="P2",
                title="Формулировка диагноза не найдена в справочнике МКБ",
                detail="Текст установленного диагноза не сопоставился с рубриками справочника.",
                evidence=text[:200],
            )
        )

    if any_code_unknown:
        unknown = [c["code"] for c in code_checks if not c["in_directory"]]
        findings.append(
            _finding(
                "B_icd_dir_code_unknown",
                severity="P2",
                title="Код МКБ отсутствует в справочнике",
                detail="Код(ы) не найдены в RU-справочнике МКБ: " + ", ".join(unknown[:6]),
                evidence=", ".join(unknown[:6]),
            )
        )

    # Правило: валидный код в справочнике + пустой/code-only текст → согласовано.
    # Mismatch только когда есть substantive free text и он плохо перекрывает title.
    substantive_text = free_text_is_substantive(text)
    if (
        any_code_in_dir
        and substantive_text
        and text_rubric_fit < thr_review
    ):
        findings.append(
            _finding(
                "B_icd_dir_text_mismatch",
                severity="P2",
                title="Формулировка диагноза слабо согласуется с рубрикой МКБ",
                detail=(
                    f"Overlap со справочником {text_rubric_fit:.2f} "
                    f"(порог review {thr_review}, ok {thr_ok})."
                ),
                evidence=text[:200],
            )
        )

    if findings:
        if any(f["code"] == "B_icd_dir_no_match" for f in findings) or any(
            f["code"] == "B_icd_dir_code_unknown" for f in findings
        ):
            verdict = "fail"
            score_pct = 35 if directory_hit else 20
        else:
            verdict = "review"
            score_pct = 60
    elif any_code_in_dir and not substantive_text:
        # Код есть в справочнике; формулировки сверх кода нет - рубрика задана кодом.
        verdict = "ok"
        score_pct = 92
    elif text_rubric_fit >= thr_ok or (directory_hit and not uniq_codes):
        verdict = "ok"
        score_pct = 95 if text_rubric_fit >= thr_ok else 85
    elif any_code_in_dir and text_rubric_fit >= thr_review:
        verdict = "review"
        score_pct = 78
    elif directory_hit:
        verdict = "ok"
        score_pct = 80
    else:
        verdict = "ok"
        score_pct = 70

    return {
        "engine": ENGINE,
        "directory_hit": directory_hit,
        "code_in_directory": any_code_in_dir if uniq_codes else None,
        "text_rubric_fit": round(text_rubric_fit, 3),
        "verdict": verdict,
        "score_pct": score_pct,
        "findings": findings,
        "candidates": candidates[:5],
        "code_checks": code_checks,
        "thresholds": {"text_fit_ok": thr_ok, "text_fit_review": thr_review},
    }


def evaluate_mo_icd_directory(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Shadow findings из case dict (слоты МО + fallback полного текста)."""
    if not icd_directory_eval_enabled() or not isinstance(case, dict):
        return []
    diag = ""
    codes: list[str] = []
    try:
        from clinical_knowledge.mo_icd_resolve import resolve_diagnosis_text_from_mo

        resolved_dx = resolve_diagnosis_text_from_mo(case)
        diag = str(resolved_dx.get("text") or "").strip()
        codes = list(resolved_dx.get("codes") or [])
        main = resolved_dx.get("main")
        if main and main not in codes:
            codes.insert(0, str(main))
    except Exception:  # noqa: BLE001
        diag = " ".join(
            str(case.get(key) or "")
            for key in (
                "clinical_diagnosis",
                "mis_diagnos",
                "diagnosis_main_text",
                "diagnosis_short",
                "diagnosis_text",
            )
            if case.get(key)
        ).strip()
        for key in ("mkb_code_main", "diagnosis_code", "icd10"):
            val = case.get(key)
            if isinstance(val, str) and val.strip():
                codes.append(val.strip().upper())
    try:
        from clinical_knowledge.mo_icd_aliases import match_query

        diag = match_query(diag) or diag
    except Exception:  # noqa: BLE001
        pass
    if not codes:
        try:
            from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo

            resolved = resolve_icd_codes_from_mo(case)
            codes = list(resolved.get("all") or [])
            main = resolved.get("main")
            if main and str(main) not in codes:
                codes.insert(0, str(main))
        except Exception:  # noqa: BLE001
            pass
    result = evaluate_diagnosis_against_icd_directory(diag, codes)
    return list(result.get("findings") or [])


def merge_icd_directory_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Добавить shadow directory findings в список (без дублей по code)."""
    out = [dict(item) for item in (findings or []) if isinstance(item, dict)]
    if not icd_directory_eval_enabled() or not case:
        return out
    existing = {str(item.get("code") or item.get("finding_code") or "") for item in out}
    try:
        shadow = evaluate_mo_icd_directory(case)
    except Exception:  # noqa: BLE001
        return out
    primary = icd_directory_primary_enabled()
    for item in shadow:
        code = str(item.get("code") or "")
        if not code or code in existing:
            continue
        row = dict(item)
        if primary:
            row["shadow"] = False
        out.append(row)
        existing.add(code)
    return out
