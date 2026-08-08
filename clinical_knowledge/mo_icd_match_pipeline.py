"""Оркестратор сверки диагноза МО со справочником МКБ (v3).

Не дублирует scorers: вызывает resolve / directory / name_match / aliases.
Чип таблицы дня - единственный источник из pipeline.chip.
"""
from __future__ import annotations

import os
from typing import Any

ENGINE = "mo_icd_match_pipeline_v3"

_DIR_CODES = {
    "B_dx_absent",
    "B_icd_dir_code_unknown",
    "B_icd_dir_no_match",
    "B_icd_dir_text_mismatch",
}
_NAME_CODES = {
    "B_icd_name_no_match",
    "B_icd_name_weak_match",
}


def icd_pipeline_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_PIPELINE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def icd_pipeline_primary_enabled() -> bool:
    """Общий primary: directory + name findings влияют на overall."""
    from clinical_knowledge.mo_icd_thresholds import pipeline_in_primary_enabled

    return pipeline_in_primary_enabled()


def normalize_mis_agreement(raw: Any) -> str:
    """Экспорт: match|partial|mismatch|unknown; legacy: 0/1/true/false."""
    agree = str(raw or "").strip().lower()
    if agree in {"1", "true", "yes", "match", "ok"}:
        return "match"
    if agree in {"0", "false", "no", "mismatch"}:
        return "mismatch"
    if agree in {"partial", "part", "stem"}:
        return "partial"
    if agree in {"unknown", "na", "n/a", "none", ""}:
        return "unknown" if agree else "skip"
    return "unknown"


def _merge_chip_status(
    *,
    diag_text: str,
    codes: list[str],
    dir_result: dict[str, Any],
    name_result: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Вернуть (chip_status, pipeline_verdict, merged_findings A-D)."""
    from clinical_knowledge.mo_icd_visit_status import status_from_finding_codes

    dir_findings = list(dir_result.get("findings") or [])
    name_findings = list(name_result.get("findings") or [])
    merged = dir_findings + name_findings

    if not (diag_text or "").strip() and not codes:
        status = status_from_finding_codes([f.get("code") for f in merged])
        if status == "unknown":
            status = "missing_dx"
        return status, "fail", merged

    dir_v = str(dir_result.get("verdict") or "")
    name_v = str(name_result.get("verdict") or "skip")
    name_fit = float(name_result.get("name_fit") or 0)
    text_fit = float(dir_result.get("text_rubric_fit") or 0)

    code_unknown = any(
        str(f.get("code") or "") in {"B_icd_dir_code_unknown", "B_icd_dir_no_match"}
        for f in dir_findings
    )
    text_mismatch = any(
        str(f.get("code") or "") == "B_icd_dir_text_mismatch" for f in dir_findings
    )

    # §3.1 приоритеты
    if code_unknown and name_v == "fail":
        return "not_in_directory", "fail", merged
    if code_unknown:
        return "not_in_directory", "review" if name_v in {"ok", "review"} else "fail", merged
    if text_mismatch or name_v == "review" or (
        name_v == "fail" and (diag_text or "").strip() and name_fit > 0
    ):
        # код ок, название слабо / противоречие текст↔код
        status = status_from_finding_codes([f.get("code") for f in merged])
        if status == "unknown":
            status = "weak_name" if (text_mismatch or name_v == "review") else "not_in_directory"
        verdict = "review" if status == "weak_name" else "fail"
        return status, verdict, merged
    if name_v == "fail" and (diag_text or "").strip() and not codes:
        return "not_in_directory", "fail", merged

    status = status_from_finding_codes([f.get("code") for f in merged])
    if status == "unknown":
        # нет негативных ICD findings
        if dir_v in {"ok", "review", "skip"} or name_v == "ok" or codes:
            status = "ok"
        else:
            from clinical_knowledge.mo_icd_thresholds import name_review, text_fit_review

            status = (
                "ok"
                if text_fit >= text_fit_review() or name_fit >= name_review()
                else "weak_name"
            )
    pipeline_verdict = "ok" if status == "ok" else ("review" if status == "weak_name" else "fail")
    return status, pipeline_verdict, merged


def evaluate_mo_icd_match(case: dict[str, Any] | None) -> dict[str, Any]:
    """Полная сверка Dx↔МКБ для одного визита."""
    from clinical_knowledge.mo_icd_aliases import expand
    from clinical_knowledge.mo_icd_directory_eval import evaluate_diagnosis_against_icd_directory
    from clinical_knowledge.mo_icd_name_match import evaluate_diagnosis_name_only
    from clinical_knowledge.mo_icd_resolve import (
        resolve_diagnosis_text_from_mo,
        resolve_icd_codes_from_mo,
    )
    from clinical_knowledge.mo_icd_visit_status import (
        chip_label_ru,
        chip_title_ru,
        status_payload,
    )

    if not isinstance(case, dict):
        chip = status_payload("unknown")
        return {
            "engine": ENGINE,
            "diag_text": "",
            "diag_source": "empty",
            "codes": [],
            "alias_expanded": "",
            "directory": {},
            "name_only": {},
            "mis_agreement": "skip",
            "chip": chip,
            "pipeline_verdict": "skip",
            "score_pct": None,
            "needs_llm_review": False,
            "findings": [],
            "seed_codes": [],
        }

    dx_info = resolve_diagnosis_text_from_mo(case)
    diag_text = str(dx_info.get("text") or "").strip()
    diag_source = str(dx_info.get("source") or "empty")

    resolved = resolve_icd_codes_from_mo(case)
    codes = [str(c) for c in (resolved.get("all") or []) if c]
    main = resolved.get("main")
    if main and str(main) not in codes:
        codes.insert(0, str(main))

    alias = expand(diag_text)
    query = str(alias.get("match_query") or diag_text).strip()
    alias_phrase = " ".join(alias.get("expanded_phrases") or [])

    dir_result = evaluate_diagnosis_against_icd_directory(query or diag_text, codes)
    if query.strip():
        name_result = evaluate_diagnosis_name_only(query)
    else:
        name_result = {
            "findings": [],
            "verdict": "skip",
            "name_fit": 0.0,
            "score_pct": None,
        }

    mis = normalize_mis_agreement(case.get("mkb_code_agreement"))

    chip_status, pipeline_verdict, ad_findings = _merge_chip_status(
        diag_text=diag_text,
        codes=codes,
        dir_result=dir_result,
        name_result=name_result,
    )
    chip = {
        "status": chip_status,
        "label_ru": chip_label_ru(chip_status),
        "title_ru": chip_title_ru(chip_status),
        "finding_codes": [
            str(f.get("code") or "")
            for f in ad_findings
            if str(f.get("code") or "") in (_DIR_CODES | _NAME_CODES)
        ],
    }

    # score: среднее доступных осей directory/name
    scores: list[float] = []
    if dir_result.get("score_pct") is not None:
        scores.append(float(dir_result["score_pct"]))
    if name_result.get("score_pct") is not None:
        scores.append(float(name_result["score_pct"]))
    score_pct = int(round(sum(scores) / len(scores))) if scores else None

    needs_llm = pipeline_verdict == "review" or any(
        str(f.get("code") or "") == "B_icd_dir_text_mismatch" for f in ad_findings
    )
    # B3: первый контакт / новый код для профиля - выше приоритет LLM-очереди
    history_queue_boost = 0
    try:
        hist = case.get("_patient_history_summary") if isinstance(case, dict) else None
        tier = str((hist or {}).get("tier") or "")
        if tier in {"first_contact", "new_for_profile"}:
            history_queue_boost = 1
            needs_llm = True
    except Exception:  # noqa: BLE001
        history_queue_boost = 0

    all_findings = list(ad_findings)
    # MIS - ось E, не в chip
    if mis == "mismatch":
        all_findings.append(
            {
                "code": "B_icd_mismatch_mis",
                "axis": "clinical_concordance",
                "severity": "P2",
                "passed": False,
                "title_ru": "Код МКБ в тексте не совпадает с диагнозом в МИС",
                "detail_ru": f"agreement={mis}",
                "evidence": (
                    f"text={resolved.get('main') or (codes[0] if codes else '')} "
                    f"vs mis={case.get('mkb_code_mis') or case.get('mis_diagnos')}"
                )[:400],
                "source_ref": "mis_data.diagnos",
                "shadow": True,
                "engine": ENGINE,
            }
        )

    return {
        "engine": ENGINE,
        "diag_text": diag_text,
        "diag_source": diag_source,
        "codes": codes,
        "alias_expanded": alias_phrase,
        "directory": dir_result,
        "name_only": name_result,
        "mis_agreement": mis,
        "chip": chip,
        "pipeline_verdict": pipeline_verdict,
        "score_pct": score_pct,
        "needs_llm_review": needs_llm,
        "history_queue_boost": history_queue_boost,
        "findings": all_findings,
        "seed_codes": list(alias.get("seed_codes") or []),
        "alias": alias,
    }


def directory_findings_from_pipeline(pipe: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        f
        for f in (pipe.get("directory") or {}).get("findings") or []
        if isinstance(f, dict)
    ]


def name_findings_from_pipeline(pipe: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        f
        for f in (pipe.get("name_only") or {}).get("findings") or []
        if isinstance(f, dict)
    ]
