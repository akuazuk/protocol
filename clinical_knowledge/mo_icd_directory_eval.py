"""Оценка диагноза МО против справочника МКБ (отдельно от подбора КП).

См. docs/plans/2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md.
Пороги text_rubric_fit: ok ≥0.35, review ≥0.25 (как consult enrichment).
Default: shadow findings; primary только при MO_ICD_DIR_IN_PRIMARY=1.
"""
from __future__ import annotations

import os
import re
from typing import Any

ENGINE = "mo_icd_directory_v1"
_SOURCE = "mo_icd_directory_v1"

# Согласовано с consult_criteria_enrichment._title_match_score
TEXT_FIT_OK = 0.35
TEXT_FIT_REVIEW = 0.25
# Минимальный lex score из suggest_icd_from_russian для directory_hit
DIR_HIT_SCORE_MIN = 0.12


def icd_directory_eval_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_DIRECTORY_EVAL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def icd_directory_primary_enabled() -> bool:
    """Merge into primary findings (affects overall). Default off = shadow."""
    raw = (os.environ.get("MO_ICD_DIR_IN_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _norm_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[а-яёa-z]{4,}", (text or "").lower()) if len(t) >= 4}


def title_match_score(diagnosis_text: str, ru_title: str | None) -> float:
    if not ru_title or not diagnosis_text:
        return 0.0
    dt = _norm_tokens(diagnosis_text)
    rt = _norm_tokens(ru_title)
    if not dt or not rt:
        return 0.0
    return len(dt & rt) / max(len(rt), 1)


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
        directory_hit = bool(candidates) and top_lex >= DIR_HIT_SCORE_MIN

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
            "thresholds": {"text_fit_ok": TEXT_FIT_OK, "text_fit_review": TEXT_FIT_REVIEW},
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

    if any_code_in_dir and text and text_rubric_fit < TEXT_FIT_REVIEW:
        findings.append(
            _finding(
                "B_icd_dir_text_mismatch",
                severity="P2",
                title="Формулировка диагноза слабо согласуется с рубрикой МКБ",
                detail=(
                    f"Overlap со справочником {text_rubric_fit:.2f} "
                    f"(порог review {TEXT_FIT_REVIEW}, ok {TEXT_FIT_OK})."
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
    elif text_rubric_fit >= TEXT_FIT_OK or (directory_hit and not uniq_codes):
        verdict = "ok"
        score_pct = 95 if text_rubric_fit >= TEXT_FIT_OK else 85
    elif any_code_in_dir and text_rubric_fit >= TEXT_FIT_REVIEW:
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
        "thresholds": {"text_fit_ok": TEXT_FIT_OK, "text_fit_review": TEXT_FIT_REVIEW},
    }


def evaluate_mo_icd_directory(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Shadow findings из case dict (слоты МО)."""
    if not icd_directory_eval_enabled() or not isinstance(case, dict):
        return []
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
    codes: list[str] = []
    try:
        from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo

        resolved = resolve_icd_codes_from_mo(case)
        codes = list(resolved.get("all") or [])
        main = resolved.get("main")
        if main and main not in codes:
            codes.insert(0, str(main))
    except Exception:  # noqa: BLE001
        for key in ("mkb_code_main", "diagnosis_code", "icd10"):
            val = case.get(key)
            if isinstance(val, str) and val.strip():
                codes.append(val.strip().upper())
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
