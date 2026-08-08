"""Оценка формулировки диагноза ↔ названия в справочнике МКБ (name_only).

Коды МКБ в тексте диагноза игнорируются (strip). См. план
docs/plans/2026-08-08-mo-icd-name-match-v2.md.
Default: shadow; primary только MO_ICD_NAME_IN_PRIMARY=1.
"""
from __future__ import annotations

import os
from typing import Any

from clinical_knowledge.clinical_text_similarity import (
    combined_score,
    normalize_for_match,
    strip_icd_codes,
    strip_leading_code_from_title,
)

ENGINE = "mo_icd_name_match_v1"
_SOURCE = "mo_icd_name_match_v1"

NAME_OK = 0.42
NAME_REVIEW = 0.28
SUGGEST_MIN = 0.08


def icd_name_match_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_NAME_MATCH") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def icd_name_match_primary_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_NAME_IN_PRIMARY") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
        "axis": "icd_name_match",
        "severity": severity,
        "passed": False,
        "title_ru": title,
        "detail_ru": detail,
        "evidence": (evidence or "")[:400],
        "source_ref": _SOURCE,
        "needs_human": False,
        "shadow": True,
        "engine": ENGINE,
        "linked_fields": ["clinical_diagnosis", "mis_diagnos"],
        "link_hint_ru": "Сверьте название диагноза с формулировкой в справочнике МКБ",
    }


def _diag_text_from_case(case: dict[str, Any]) -> str:
    return " ".join(
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


def _suggest_candidates(diag_text: str, *, max_results: int = 12) -> list[dict[str, Any]]:
    import icd_mkb

    try:
        rows = icd_mkb.suggest_icd_from_russian(diag_text, max_results=max_results)
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title_ru") or "").strip()
        if not title:
            continue
        out.append(
            {
                "code": row.get("code"),
                "title_ru": title,
                "lex_score": float(row.get("score") or 0),
                "match_method": row.get("match_method"),
            }
        )
    return out


def evaluate_diagnosis_name_only(diag_text: str) -> dict[str, Any]:
    """Сверка названия диагноза со справочником (без равенства кодов)."""
    raw = (diag_text or "").strip()
    cleaned = strip_icd_codes(raw)
    norm = normalize_for_match(cleaned)
    empty = {
        "directory_name_hit": False,
        "best_code": None,
        "best_title_ru": None,
        "name_fit": 0.0,
        "verdict": "fail",
        "score_pct": 0,
        "findings": [],
        "candidates": [],
        "similarity": None,
        "thresholds": {"name_ok": NAME_OK, "name_review": NAME_REVIEW},
        "engine": ENGINE,
    }
    if len(norm) < 3:
        empty["findings"] = [
            _finding(
                "B_icd_name_no_match",
                severity="P2",
                title="Название диагноза не сопоставлено со справочником МКБ",
                detail="Слишком короткий или пустой текст диагноза после удаления кодов",
                evidence=raw[:200],
            )
        ]
        return empty

    candidates = _suggest_candidates(cleaned or raw)
    # Пересчёт по очищенным названиям (не lex_score и не равенство кодов)
    scored: list[dict[str, Any]] = []
    for row in candidates:
        title_clean = strip_leading_code_from_title(str(row.get("title_ru") or ""))
        sim = combined_score(cleaned, title_clean)
        scored.append(
            {
                "code": row.get("code"),
                "title_ru": row.get("title_ru"),
                "title_ru_clean": title_clean,
                "lex_score": row.get("lex_score"),
                "similarity": sim,
                "score": sim["combined"],
            }
        )
    scored.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    best = scored[0] if scored else None
    # Если suggest пуст или слабый - best_match на том же списке (noop) /
    # дополнительно: если есть хоть один кандидат с lex, всё равно name score решает.
    if best is None:
        findings = [
            _finding(
                "B_icd_name_no_match",
                severity="P2",
                title="Название диагноза не найдено в справочнике МКБ",
                detail="Нет лексических кандидатов по формулировке (коды не учитывались)",
                evidence=raw[:200],
            )
        ]
        empty["findings"] = findings
        return empty

    name_fit = float(best["score"] or 0)
    # Слабый lex без name fit не считаем hit
    if name_fit < NAME_REVIEW and float(best.get("lex_score") or 0) < SUGGEST_MIN:
        name_fit = float(best["score"] or 0)

    if name_fit >= NAME_OK:
        verdict, score_pct, findings = "ok", int(round(min(100.0, name_fit * 100))), []
    elif name_fit >= NAME_REVIEW:
        verdict, score_pct = "review", int(round(min(100.0, name_fit * 100)))
        findings = [
            _finding(
                "B_icd_name_weak_match",
                severity="P3",
                title="Формулировка диагноза слабо совпадает со справочником МКБ",
                detail=(
                    f"Ближе всего: {best.get('title_ru_clean') or best.get('title_ru')} "
                    f"({best.get('code')}), score={name_fit:.2f}"
                ),
                evidence=raw[:200],
            )
        ]
    else:
        verdict, score_pct = "fail", int(round(max(0.0, name_fit * 100)))
        findings = [
            _finding(
                "B_icd_name_no_match",
                severity="P2",
                title="Название диагноза не сопоставлено со справочником МКБ",
                detail=(
                    f"Лучший кандидат: {best.get('title_ru_clean') or best.get('title_ru')} "
                    f"({best.get('code')}), score={name_fit:.2f} (коды не учитывались)"
                ),
                evidence=raw[:200],
            )
        ]

    return {
        "directory_name_hit": name_fit >= NAME_REVIEW,
        "best_code": best.get("code"),
        "best_title_ru": best.get("title_ru_clean") or best.get("title_ru"),
        "name_fit": round(name_fit, 3),
        "verdict": verdict,
        "score_pct": score_pct,
        "findings": findings,
        "candidates": scored[:5],
        "similarity": best.get("similarity"),
        "thresholds": {"name_ok": NAME_OK, "name_review": NAME_REVIEW},
        "engine": ENGINE,
    }


def evaluate_mo_icd_name_match(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not icd_name_match_enabled() or not isinstance(case, dict):
        return []
    result = evaluate_diagnosis_name_only(_diag_text_from_case(case))
    return list(result.get("findings") or [])


def merge_icd_name_match_into_findings(
    findings: list[dict[str, Any]] | None,
    case: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out = [dict(item) for item in (findings or []) if isinstance(item, dict)]
    if not icd_name_match_enabled() or not case:
        return out
    existing = {str(item.get("code") or item.get("finding_code") or "") for item in out}
    try:
        shadow = evaluate_mo_icd_name_match(case)
    except Exception:  # noqa: BLE001
        return out
    primary = icd_name_match_primary_enabled()
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


# re-export for callers / future section-align
__all__ = [
    "ENGINE",
    "NAME_OK",
    "NAME_REVIEW",
    "evaluate_diagnosis_name_only",
    "evaluate_mo_icd_name_match",
    "icd_name_match_enabled",
    "icd_name_match_primary_enabled",
    "merge_icd_name_match_into_findings",
]
