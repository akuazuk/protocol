"""LLM-оценка результата анализа КЗ для кабинета методиста (этап 2 после детерминированного разбора)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from .feedback_store import (
    _VALID_KZ_COMPLIANCE_GOLD,
    _VALID_TAGS,
    _VALID_VERDICTS,
)
from .methodist_context import STRUCTURED_BLOCK_ROWS, build_methodist_review_context

SYSTEM_METHODIST_AI_REVIEW = """Ты методист-врач методслужбы. Тебе передали:
1) текст консультативного заключения (КЗ);
2) результат автоматической проверки системы (8 блоков, правила протоколов, протоколы, при L2 — критерии LLM).

Задача — НЕ перепроверять клинику с нуля, а:
- оценить, насколько само КЗ соответствует клиническим протоколам Минздрава РБ (экспертная gold-метка);
- оценить, насколько ВЕРНО система оценила это КЗ (рейтинг и вердикт точности системы);
- перечислить, что врачу улучшить в тексте КЗ;
- указать, где система ошиблась (блоки/правила/RAG), если есть расхождения.

Строгие правила:
- Не выдавай юридических или МЭЭ-вердиктов.
- Опирайся на переданные протоколы и правила; если данных мало — insufficient_data и понизь confidence.
- block_key только из списка ключей блоков ниже; verdict: agree | disagree (disagree только если системный % явно неверен).
- rule_overrides только для rule_id из списка findings; human_pass: true/false/null (null = не применимо).
- tags только из: wrong_protocol, missed_protocol, false_positive_rule, missed_issue, wrong_population, wrong_diagnosis_block, wrong_treatment_block, score_misleading, other.
- improvements_ru — конкретные правки для врача (3–7 пунктов), не общие фразы.

Верни ОДИН JSON (без markdown):
{
  "kz_compliance_gold": "compliant|mostly_compliant|partially_compliant|non_compliant|insufficient_data",
  "system_accuracy_rating": <целое 1-5>,
  "system_accuracy_verdict": "correct|mostly_correct|partially_wrong|wrong",
  "tags": ["..."],
  "summary_ru": "<2-4 предложения: итог по КЗ и по работе системы>",
  "improvements_ru": ["<что улучшить в КЗ>"],
  "system_notes_ru": "<где система ошиблась или что сделала хорошо>",
  "block_overrides": [{"block_key": "<key>", "verdict": "agree|disagree", "note": "<до 200 символов>"}],
  "rule_overrides": [{"rule_id": "<id>", "human_pass": true|false, "note": "<до 200 символов>"}],
  "retrieval_fix": {"rejected_path": "<из топ протоколов или пусто>", "chosen_path": "<правильный путь или пусто>", "note": ""},
  "confidence": "high|medium|low"
}"""

_VALID_BLOCK_KEYS = frozenset(row["key"] for row in STRUCTURED_BLOCK_ROWS)
_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})


def methodist_ai_review_enabled() -> bool:
    return os.environ.get("METHODIST_AI_REVIEW", "1").strip().lower() in ("1", "true", "yes", "on")


def _build_prompt(result: dict[str, Any], full_text: str) -> str:
    ctx = build_methodist_review_context(result, full_text)
    comp = ctx.get("compliance") or {}
    blocks = comp.get("blocks") or []
    findings = ctx.get("rules_findings") or []
    llm_crit = ctx.get("llm_criteria") or []
    paths = ctx.get("retrieval_top_paths") or ctx.get("matched_protocol_paths") or []

    block_lines = [
        f"- {b.get('label_ru') or b.get('key')}: {b.get('score_pct')}% (key={b.get('key')})"
        for b in blocks
        if b.get("score_pct") is not None
    ]
    finding_lines = [
        f"- {f.get('rule_id')}: {'OK' if f.get('passed') else 'FAIL'} — {(f.get('title_ru') or f.get('message_ru') or '')[:120]}"
        for f in findings[:24]
    ]
    llm_lines = [
        f"- {c.get('name_ru')}: {c.get('score_pct')}%"
        for c in llm_crit[:12]
        if c.get("score_pct") is not None
    ]

    kz_display = (ctx.get("kz_text_display") or full_text or "")[:12000]
    rev = result.get("review") or {}
    summary = (rev.get("summary_ru") or "")[:800]

    parts = [
        SYSTEM_METHODIST_AI_REVIEW,
        "\n\n--- КЛЮЧИ БЛОКОВ (block_key): ",
        ", ".join(sorted(_VALID_BLOCK_KEYS)),
        "\n\n--- ТЕКСТ КЗ ---\n",
        kz_display,
        "\n\n--- ОЦЕНКА СИСТЕМЫ ---",
        f"overall: {comp.get('overall_pct')}%",
        f"structured: {comp.get('structured_pct')}%",
        f"rules: {comp.get('rules_pct')}%",
        f"status: {comp.get('overall_status') or '—'}",
    ]
    if summary:
        parts.extend(["\nsummary_ru системы: ", summary])
    if block_lines:
        parts.extend(["\n\n8 блоков:\n", "\n".join(block_lines)])
    if finding_lines:
        parts.extend(["\n\nПравила протокола:\n", "\n".join(finding_lines)])
    if llm_lines:
        parts.extend(["\n\nКритерии LLM (L2):\n", "\n".join(llm_lines)])
    if paths:
        parts.extend(["\n\nТоп протоколов:\n", "\n".join(f"- {p}" for p in paths[:8])])
    return "".join(parts)


def _clamp_rating(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def normalize_ai_review(raw: dict[str, Any]) -> dict[str, Any]:
    """Валидация и нормализация ответа модели."""
    if not isinstance(raw, dict):
        raise ValueError("Ответ модели не является JSON-объектом")

    gold = str(raw.get("kz_compliance_gold") or "").strip()
    if gold not in _VALID_KZ_COMPLIANCE_GOLD:
        raise ValueError(f"Неверный kz_compliance_gold: {gold or 'пусто'}")

    verdict = str(raw.get("system_accuracy_verdict") or "").strip()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"Неверный system_accuracy_verdict: {verdict or 'пусто'}")

    rating = _clamp_rating(raw.get("system_accuracy_rating"))
    if rating is None:
        raise ValueError("system_accuracy_rating должен быть 1–5")

    tags_in = raw.get("tags") or []
    if not isinstance(tags_in, list):
        tags_in = []
    tags = [t for t in (str(x).strip() for x in tags_in) if t in _VALID_TAGS]

    improvements = raw.get("improvements_ru") or []
    if isinstance(improvements, str):
        improvements = [improvements]
    if not isinstance(improvements, list):
        improvements = []
    improvements_ru = [str(x).strip() for x in improvements if str(x).strip()][:10]

    block_overrides: list[dict[str, Any]] = []
    for bo in raw.get("block_overrides") or []:
        if not isinstance(bo, dict):
            continue
        bk = str(bo.get("block_key") or "").strip()
        vd = str(bo.get("verdict") or "").strip()
        if bk not in _VALID_BLOCK_KEYS or vd not in ("agree", "disagree"):
            continue
        if vd != "disagree":
            continue
        block_overrides.append(
            {
                "block_key": bk,
                "verdict": vd,
                "note": str(bo.get("note") or "")[:280],
            }
        )

    rule_overrides: list[dict[str, Any]] = []
    for ro in raw.get("rule_overrides") or []:
        if not isinstance(ro, dict):
            continue
        rid = str(ro.get("rule_id") or "").strip()
        if not rid:
            continue
        hp = ro.get("human_pass")
        if hp is not None and not isinstance(hp, bool):
            continue
        rule_overrides.append(
            {
                "rule_id": rid,
                "human_pass": hp,
                "note": str(ro.get("note") or "")[:280],
            }
        )

    retrieval_fix = None
    rf = raw.get("retrieval_fix")
    if isinstance(rf, dict):
        chosen = str(rf.get("chosen_path") or "").strip()
        rejected = str(rf.get("rejected_path") or "").strip()
        if chosen:
            retrieval_fix = {
                "rejected_path": rejected,
                "chosen_path": chosen,
                "note": str(rf.get("note") or "")[:280],
            }

    conf = str(raw.get("confidence") or "medium").strip().lower()
    if conf not in _VALID_CONFIDENCE:
        conf = "medium"

    return {
        "kz_compliance_gold": gold,
        "system_accuracy_rating": rating,
        "system_accuracy_verdict": verdict,
        "tags": tags,
        "summary_ru": str(raw.get("summary_ru") or "").strip()[:2000],
        "improvements_ru": improvements_ru,
        "system_notes_ru": str(raw.get("system_notes_ru") or "").strip()[:2000],
        "block_overrides": block_overrides,
        "rule_overrides": rule_overrides,
        "retrieval_fix": retrieval_fix,
        "confidence": conf,
        "review_source": "ai_assisted",
    }


def _try_parse_json(t: str) -> dict[str, Any] | None:
    if not t:
        return None
    s = t.strip()
    if "```" in s:
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.M)
        s = re.sub(r"\s*```\s*$", "", s, flags=re.M)
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def run_methodist_ai_review(
    result: dict[str, Any],
    full_text: str,
    *,
    generate_fn: Callable[..., Any] | None = None,
    extract_text_fn: Callable[[Any], str] | None = None,
    parse_json_fn: Callable[[str], dict | None] | None = None,
    get_model_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Вызов LLM для предзаполнения оценки методиста."""
    if generate_fn is None or extract_text_fn is None or get_model_fn is None:
        import rag_server as rs

        generate_fn = generate_fn or rs.generate_gemini_methodist_ai_review
        extract_text_fn = extract_text_fn or rs._extract_gemini_text
        parse_json_fn = parse_json_fn or rs._try_parse_json
        get_model_fn = get_model_fn or rs.get_methodist_gemini

    model = get_model_fn()
    prompt = _build_prompt(result, full_text)
    resp = generate_fn(model, prompt)
    txt = extract_text_fn(resp)
    parsed = parse_json_fn(txt) if parse_json_fn else _try_parse_json(txt)
    if not parsed:
        raise ValueError("Модель не вернула корректный JSON для оценки методиста")
    normalized = normalize_ai_review(parsed)
    model_name = os.environ.get("GEMINI_METHODIST_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    normalized["model_used"] = model_name
    return normalized
