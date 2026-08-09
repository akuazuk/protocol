"""Оценка формулировки диагноза ↔ названия в справочнике МКБ (name_only).

Коды МКБ в тексте диагноза игнорируются (strip). См. план
docs/plans/2026-08-08-mo-icd-name-match-v2.md.
Default: shadow; primary только MO_ICD_NAME_IN_PRIMARY=1.
"""
from __future__ import annotations

import os
from typing import Any

from clinical_knowledge.clinical_text_similarity import (
    best_combined_against_title,
    normalize_for_match,
    strip_icd_codes,
    strip_leading_code_from_title,
)

from clinical_knowledge.mo_icd_thresholds import (  # noqa: E402
    NAME_OK as NAME_OK_DEFAULT,
    NAME_REVIEW as NAME_REVIEW_DEFAULT,
    SUGGEST_MIN as SUGGEST_MIN_DEFAULT,
    name_ok as _name_ok,
    name_review as _name_review,
    pipeline_in_primary_enabled,
    suggest_min as _suggest_min,
)

ENGINE = "mo_icd_name_match_v1"
_SOURCE = "mo_icd_name_match_v1"

# Совместимость тестов / импортов: дефолты; runtime - через getters ниже.
NAME_OK = NAME_OK_DEFAULT
NAME_REVIEW = NAME_REVIEW_DEFAULT
SUGGEST_MIN = SUGGEST_MIN_DEFAULT


def icd_name_match_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_NAME_MATCH") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def icd_name_match_primary_enabled() -> bool:
    if pipeline_in_primary_enabled():
        return True
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
    try:
        from clinical_knowledge.mo_icd_resolve import resolve_diagnosis_text_from_mo

        return str(resolve_diagnosis_text_from_mo(case).get("text") or "").strip()
    except Exception:  # noqa: BLE001
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


def _suggest_candidates(
    diag_text: str,
    *,
    raw_for_codes: str | None = None,
    max_results: int = 12,
) -> list[dict[str, Any]]:
    import icd_mkb

    try:
        rows = icd_mkb.suggest_icd_from_russian(diag_text, max_results=max_results)
    except Exception:  # noqa: BLE001
        rows = []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(row: dict[str, Any]) -> None:
        title = str(row.get("title_ru") or "").strip()
        code = str(row.get("code") or "").strip().upper()
        key = code or title.lower()
        if not title or key in seen:
            return
        seen.add(key)
        out.append(
            {
                "code": row.get("code"),
                "title_ru": title,
                "lex_score": float(row.get("score") or 0),
                "match_method": row.get("match_method") or "suggest",
            }
        )

    for row in rows or []:
        if isinstance(row, dict):
            _add(row)

    # Сид из кодов в исходном тексте (до strip): name_only не сравнивает коды,
    # но title по коду нужен, иначе «J45 Бронхиальная астма…» теряется.
    try:
        scanned = icd_mkb.normalize_text_for_icd_scan(raw_for_codes or diag_text or "")
        for match in icd_mkb.ICD10_CODE_RE.finditer(scanned or ""):
            code = icd_mkb.normalize_icd_code(match.group(1))
            if not code or not icd_mkb.is_code_in_ru_reference(code):
                continue
            title = icd_mkb.ru_title(code) or ""
            _add(
                {
                    "code": code,
                    "title_ru": title,
                    "score": 1.0,
                    "match_method": "code_seed",
                }
            )
            cat = code.split(".", 1)[0]
            if cat != code and icd_mkb.is_code_in_ru_reference(cat):
                _add(
                    {
                        "code": cat,
                        "title_ru": icd_mkb.ru_title(cat) or "",
                        "score": 0.9,
                        "match_method": "code_seed_category",
                    }
                )
            elif cat == code and icd_mkb.is_code_in_ru_reference(cat):
                # уже категория; дополнительно типичная подрубрика .9
                soft = f"{cat}.9"
                if icd_mkb.is_code_in_ru_reference(soft):
                    _add(
                        {
                            "code": soft,
                            "title_ru": icd_mkb.ru_title(soft) or "",
                            "score": 0.85,
                            "match_method": "code_seed_soft",
                        }
                    )
    except Exception:  # noqa: BLE001
        pass
    return out


def evaluate_diagnosis_name_only(
    diag_text: str,
    *,
    history_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Сверка названия диагноза со справочником (без равенства кодов)."""
    raw = (diag_text or "").strip()
    cleaned = strip_icd_codes(raw)
    norm = normalize_for_match(cleaned)
    try:
        from clinical_knowledge.mo_patient_history_bundle import name_match_threshold_delta

        thr_delta = float(name_match_threshold_delta(history_summary))
    except Exception:  # noqa: BLE001
        thr_delta = 0.0
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
        "thresholds": {
            "name_ok": _name_ok(),
            "name_review": max(0.05, _name_review() + thr_delta),
        },
        "engine": ENGINE,
        "history_threshold_delta": thr_delta,
    }
    if len(norm) < 3:
        # Пустой/короткий текст без названия - зона B_dx_absent (directory), не дублируем
        empty["verdict"] = "skip"
        empty["score_pct"] = None
        return empty

    candidates = _suggest_candidates(cleaned or raw, raw_for_codes=raw)
    # Пересчёт по фразам мультидиагноза (не lex_score и не равенство кодов)
    scored: list[dict[str, Any]] = []
    for row in candidates:
        title_clean = strip_leading_code_from_title(str(row.get("title_ru") or ""))
        sim = best_combined_against_title(cleaned, title_clean)
        scored.append(
            {
                "code": row.get("code"),
                "title_ru": row.get("title_ru"),
                "title_ru_clean": title_clean,
                "lex_score": row.get("lex_score"),
                "similarity": sim,
                "matched_phrase": sim.get("matched_phrase") or "",
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
    thr_ok = _name_ok()
    thr_review = max(0.05, _name_review() + thr_delta)
    thr_suggest = _suggest_min()
    # Слабый lex без name fit не считаем hit
    if name_fit < thr_review and float(best.get("lex_score") or 0) < thr_suggest:
        name_fit = float(best["score"] or 0)

    if name_fit >= thr_ok:
        verdict, score_pct, findings = "ok", int(round(min(100.0, name_fit * 100))), []
    elif name_fit >= thr_review:
        verdict, score_pct = "review", int(round(min(100.0, name_fit * 100)))
        findings = [
            _finding(
                "B_icd_name_weak_match",
                severity="P3",
                title="Формулировка диагноза слабо совпадает со справочником МКБ",
                detail=(
                    f"Ближе всего: {best.get('title_ru_clean') or best.get('title_ru')} "
                    f"({best.get('code')}), score={name_fit:.2f}"
                    + (
                        f"; фраза: {best.get('matched_phrase')}"
                        if best.get("matched_phrase")
                        else ""
                    )
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
                    + (
                        f"; фраза: {best.get('matched_phrase')}"
                        if best.get("matched_phrase")
                        else ""
                    )
                ),
                evidence=raw[:200],
            )
        ]

    return {
        "directory_name_hit": name_fit >= thr_review,
        "best_code": best.get("code"),
        "best_title_ru": best.get("title_ru_clean") or best.get("title_ru"),
        "name_fit": round(name_fit, 3),
        "verdict": verdict,
        "score_pct": score_pct,
        "findings": findings,
        "candidates": scored[:5],
        "similarity": best.get("similarity"),
        "thresholds": {"name_ok": thr_ok, "name_review": thr_review},
        "engine": ENGINE,
    }


def evaluate_mo_icd_name_match(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not icd_name_match_enabled() or not isinstance(case, dict):
        return []
    text = _diag_text_from_case(case)
    try:
        from clinical_knowledge.mo_icd_aliases import match_query

        text = match_query(text) or text
    except Exception:  # noqa: BLE001
        pass
    # B0: resolve_diagnosis_text_from_mo уже даёт near_code fallback
    hist = case.get("_patient_history_summary")
    if not isinstance(hist, dict):
        hist = None
        try:
            from clinical_knowledge.mo_patient_history_bundle import (
                attach_bundle_to_case,
                history_summary_for_analyzers,
                patient_history_enabled,
            )

            if patient_history_enabled() and not case.get("_patient_history"):
                attach_bundle_to_case(case)
            hist = history_summary_for_analyzers(case.get("_patient_history"))
        except Exception:  # noqa: BLE001
            hist = None
    result = evaluate_diagnosis_name_only(text, history_summary=hist)
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
