"""LLM-судья серой зоны Dx↔МКБ (фаза 4).

Только когда pipeline.needs_llm_review. Не вызывает Gemini сам по себе -
generate_fn передаёт CLI/night на GCE. Default off: MO_ICD_LLM_REVIEW=0.
Findings всегда shadow; в overall не входят.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from clinical_knowledge.mo_llm_action_judge import extract_json_object

ENGINE = "mo_icd_llm_review_v1"
_SOURCE = "mo_icd_llm_review_v1"
SCHEMA_VERSION = 1

AGREE_VALUES = frozenset({"yes", "partial", "no"})

FINDING_BY_AGREE = {
    "yes": "B_icd_llm_review_yes",
    "partial": "B_icd_llm_review_partial",
    "no": "B_icd_llm_review_no",
}

TITLE_BY_AGREE = {
    "yes": "LLM: формулировка согласуется с кодом/рубрикой МКБ",
    "partial": "LLM: частичное согласие формулировки с МКБ",
    "no": "LLM: формулировка не согласуется с кодом/рубрикой МКБ",
}


def icd_llm_review_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_LLM_REVIEW") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def icd_llm_clear_weak_enabled() -> bool:
    """Снимать weak name findings при agree=yes. Default off."""
    raw = (os.environ.get("MO_ICD_LLM_CLEAR_WEAK") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _clip(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def build_llm_review_pack(pipe: dict[str, Any] | None) -> dict[str, Any] | None:
    """Собрать вход для judge из результата evaluate_mo_icd_match."""
    if not isinstance(pipe, dict):
        return None
    if not pipe.get("needs_llm_review"):
        return None
    diag = str(pipe.get("diag_text") or "").strip()
    codes = [str(c) for c in (pipe.get("codes") or []) if c]
    code = codes[0] if codes else ""
    title = ""
    if code:
        try:
            import icd_mkb

            title = str(icd_mkb.ru_title(code) or "").strip()
        except Exception:  # noqa: BLE001
            title = ""
    candidates: list[dict[str, Any]] = []
    for row in (pipe.get("name_only") or {}).get("candidates") or []:
        if not isinstance(row, dict):
            continue
        candidates.append(
            {
                "code": row.get("code"),
                "title_ru": _clip(str(row.get("title_ru_clean") or row.get("title_ru") or ""), 120),
                "score": row.get("score") or row.get("name_fit"),
            }
        )
        if len(candidates) >= 3:
            break
    if not diag and not code:
        return None
    return {
        "diag_text": _clip(diag, 400),
        "code": code,
        "ru_title": _clip(title, 200),
        "candidates": candidates,
        "pipeline_verdict": str(pipe.get("pipeline_verdict") or ""),
        "chip_status": str((pipe.get("chip") or {}).get("status") or ""),
        "has_text_mismatch": any(
            str(f.get("code") or "") == "B_icd_dir_text_mismatch"
            for f in (pipe.get("findings") or [])
            if isinstance(f, dict)
        ),
    }


def build_prompt(pack: dict[str, Any]) -> str:
    cands = pack.get("candidates") or []
    cand_lines = []
    for i, c in enumerate(cands, 1):
        cand_lines.append(
            f"{i}. {c.get('code')}: {c.get('title_ru')} (score={c.get('score')})"
        )
    cand_block = "\n".join(cand_lines) if cand_lines else "(нет кандидатов)"
    return (
        "Ты методист по МКБ-10 (RU). Оцени, согласуется ли формулировка диагноза "
        "с указанным кодом и/или кандидатами справочника.\n"
        "Ответь ТОЛЬКО JSON-объектом без markdown:\n"
        '{"agree":"yes|partial|no","reason_ru":"≤160 символов","suggested_code":"K29.3 или null"}\n\n'
        f"diag_text: {pack.get('diag_text')}\n"
        f"code: {pack.get('code') or 'null'}\n"
        f"ru_title(code): {pack.get('ru_title') or 'null'}\n"
        f"pipeline_verdict: {pack.get('pipeline_verdict')}\n"
        f"text_mismatch: {pack.get('has_text_mismatch')}\n"
        f"top name candidates:\n{cand_block}\n"
    )


def validate_llm_review(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("корень должен быть объектом")
    agree = str(raw.get("agree") or "").strip().lower()
    aliases = {"y": "yes", "true": "yes", "ok": "yes", "n": "no", "false": "no"}
    agree = aliases.get(agree, agree)
    if agree not in AGREE_VALUES:
        raise ValueError(f"agree недопустим: {agree}")
    reason = _clip(str(raw.get("reason_ru") or ""), 160)
    if not reason:
        raise ValueError("reason_ru обязателен")
    sug = raw.get("suggested_code")
    if sug is None or str(sug).strip().lower() in {"", "null", "none"}:
        suggested: str | None = None
    else:
        suggested = str(sug).strip().upper().replace(" ", "")
        if not re.match(r"^[A-TV-Z]\d{2}(?:\.\d{1,4})?$", suggested, re.I):
            raise ValueError(f"suggested_code формат: {suggested}")
    return {
        "agree": agree,
        "reason_ru": reason,
        "suggested_code": suggested,
        "engine": ENGINE,
        "schema_version": SCHEMA_VERSION,
    }


def findings_from_review(
    validated: dict[str, Any],
    *,
    pack: dict[str, Any],
) -> list[dict[str, Any]]:
    agree = str(validated.get("agree") or "")
    code = FINDING_BY_AGREE.get(agree)
    if not code:
        return []
    severity = "P3" if agree == "yes" else ("P3" if agree == "partial" else "P2")
    sug = validated.get("suggested_code")
    detail = validated.get("reason_ru") or ""
    if sug:
        detail = f"{detail}; suggested={sug}"
    return [
        {
            "code": code,
            "axis": "icd_llm_review",
            "severity": severity,
            "passed": agree == "yes",
            "title_ru": TITLE_BY_AGREE[agree],
            "detail_ru": _clip(str(detail), 300),
            "evidence": _clip(
                f"dx={pack.get('diag_text')}; code={pack.get('code')}",
                400,
            ),
            "source_ref": _SOURCE,
            "needs_human": agree != "yes",
            "shadow": True,
            "engine": ENGINE,
            "linked_fields": ["clinical_diagnosis", "mkb_code_main"],
            "link_hint_ru": "Проверьте согласие формулировки диагноза с МКБ (LLM)",
            "llm_agree": agree,
            "suggested_code": sug,
        }
    ]


def apply_clear_weak_findings(
    findings: list[dict[str, Any]],
    validated: dict[str, Any],
) -> list[dict[str, Any]]:
    """Опционально убрать weak name findings при agree=yes."""
    if not icd_llm_clear_weak_enabled():
        return list(findings)
    if str(validated.get("agree") or "") != "yes":
        return list(findings)
    drop = {"B_icd_name_weak_match", "B_icd_dir_text_mismatch"}
    return [f for f in findings if str(f.get("code") or "") not in drop]


def review_one(
    pipe: dict[str, Any],
    *,
    generate_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Один визит: pack → prompt → JSON → findings. Без generate_fn не зовёт сеть."""
    empty = {
        "engine": ENGINE,
        "skipped": True,
        "reason": "not_needed",
        "review": None,
        "findings": [],
        "pack": None,
    }
    if not icd_llm_review_enabled():
        empty["reason"] = "flag_off"
        return empty
    pack = build_llm_review_pack(pipe)
    if not pack:
        return empty
    if generate_fn is None:
        empty["reason"] = "no_generate_fn"
        empty["pack"] = pack
        empty["skipped"] = True
        return empty
    prompt = build_prompt(pack)
    raw_text = generate_fn(prompt)
    parsed = extract_json_object(raw_text)
    validated = validate_llm_review(parsed)
    findings = findings_from_review(validated, pack=pack)
    return {
        "engine": ENGINE,
        "skipped": False,
        "reason": "ok",
        "review": validated,
        "findings": findings,
        "pack": pack,
        "prompt_chars": len(prompt),
    }
