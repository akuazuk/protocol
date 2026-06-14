"""LLM-оценка выдачи поиска протоколов для кабинета методиста (этап 2 после RAG+ranking)."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

SEARCH_VALID_TAGS = frozenset({
    "wrong_protocol",
    "missed_protocol",
    "wrong_population",
    "query_too_vague",
    "wrong_rubric",
    "wrong_condition",
    "wrong_section",
    "wrong_icd_suggestion",
    "other",
})

_TAG_TO_FUNNEL_STEP = {
    "query_too_vague": 0,
    "wrong_population": 1,
    "wrong_icd_suggestion": 2,
    "wrong_rubric": 3,
    "wrong_protocol": 4,
    "missed_protocol": 4,
    "wrong_condition": 5,
    "wrong_section": 6,
}


def suggested_funnel_step_from_tags(tags: list[str]) -> int | None:
    for tag in tags:
        if tag in _TAG_TO_FUNNEL_STEP:
            return _TAG_TO_FUNNEL_STEP[tag]
    return None

SYSTEM_METHODIST_SEARCH_AI_REVIEW = """Ты методист-врач методслужбы и аудитор качества поиска клинических протоколов Минздрава РБ в ПО «Protocol».

Главная задача — МЕТА-ОЦЕНКА: насколько ВЕРНО система подобрала и ранжировала PDF-протоколы по запросу врача.
Это НЕ установление диагноза и НЕ клиническая консультация.

Вход:
1) текст запроса (диагноз / МКБ / жалобы);
2) top протоколы из RAG + оценки модели (confidence, match_reason);
3) список retrieval (BM25 + rerank).

Что нужно сделать (в порядке приоритета):
1) ranking_verdict и ranking_rating (1-5) — насколько верен top-1 и порядок выдачи.
2) engine_improvements_ru — 3-7 конкретных правок для ДВИЖКА поиска в проекте (RAG, embed rerank, routing, dedup, ICD pipeline).
   Примеры: «при I10 поднимать кардиологический КП», «отфильтровать детский протокол», «усилить match по коду K64».
3) retrieval_fix — если top-1 неверен или нужный КП не в top-3: rejected_path (ошибочный top) и chosen_path (правильный PDF из каталога или пусто если неизвестен).
4) tags — wrong_protocol, missed_protocol, wrong_population, query_too_vague, wrong_rubric, wrong_condition, wrong_section, wrong_icd_suggestion, other.
5) top1_relevant — true если первый протокол клинически уместен для запроса (даже если порядок остальных неверен).
6) suggested_funnel_step — 0–7, на каком шаге воронки вероятнее всего ошибка (если есть).

Если выдача полностью верна (top-1 релевантен, порядок разумный): ranking_verdict=correct, retrieval_fix=null, tags=[].

Запрещено:
- Рекомендации лечения пациенту.
- Выдумывать пути PDF вне списка candidates/chosen_hint.

Верни ОДИН JSON (без markdown):
{
  "ranking_verdict": "correct|mostly_correct|partially_wrong|wrong",
  "ranking_rating": <1-5>,
  "summary_ru": "<2-4 предложения о качестве подбора>",
  "engine_improvements_ru": ["<правка для RAG/поиска Protocol>"],
  "system_notes_ru": "<где ошиблась система: top, пропуск, популяция>",
  "tags": ["..."],
  "top1_relevant": true|false,
  "suggested_funnel_step": <0-7|null>,
  "retrieval_fix": {"rejected_path": "", "chosen_path": "", "note": ""} | null,
  "confidence": "high|medium|low"
}"""

_VALID_VERDICTS = frozenset({"correct", "mostly_correct", "partially_wrong", "wrong"})
_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})


def _basename(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.rsplit("/", 1)[-1] if p else ""


def _build_prompt(assist_payload: dict[str, Any]) -> str:
    query = (assist_payload.get("query") or "").strip()[:4000]
    llm = assist_payload.get("llm_json") or {}
    protos = llm.get("protocols") or []
    retrieval = assist_payload.get("retrieval") or []
    icd = assist_payload.get("icd_codes") or assist_payload.get("icd_detected") or []

    proto_lines: list[str] = []
    for i, pr in enumerate(protos[:8]):
        if not isinstance(pr, dict):
            continue
        path = str(pr.get("path") or "")
        conf = pr.get("confidence_score")
        reason = str(pr.get("match_reason") or "")[:240]
        proto_lines.append(
            f"{i + 1}. {_basename(path)} | path={path} | conf={conf} | {reason}"
        )

    ret_lines: list[str] = []
    seen: set[str] = set()
    for r in retrieval[:12]:
        if not isinstance(r, dict):
            continue
        path = str(r.get("path") or "")
        base = _basename(path).lower()
        if base and base in seen:
            continue
        if base:
            seen.add(base)
        score = r.get("score") or r.get("rag_score")
        ret_lines.append(f"- {_basename(path)} | path={path} | score={score}")

    icd_str = ", ".join(str(c) for c in (icd or [])[:8]) if icd else "не указаны"

    parts = [
        SYSTEM_METHODIST_SEARCH_AI_REVIEW,
        "\n\n--- ЗАПРОС ---\n",
        query or "(пусто)",
        "\n\n--- МКБ (если есть) ---\n",
        icd_str,
        "\n\n--- TOP ПРОТОКОЛЫ (выдача) ---\n",
        "\n".join(proto_lines) if proto_lines else "(нет)",
        "\n\n--- RETRIEVAL (кандидаты) ---\n",
        "\n".join(ret_lines) if ret_lines else "(нет)",
        "\n\n--- КАНДИДАТЫ для chosen_path (только из списков выше) ---",
    ]
    candidates = list(dict.fromkeys(
        [str(p.get("path") or "") for p in protos if isinstance(p, dict)]
        + [str(r.get("path") or "") for r in retrieval if isinstance(r, dict)]
    ))
    candidates = [c for c in candidates if c.strip()][:16]
    parts.append("\n" + "\n".join(f"- {c}" for c in candidates) if candidates else "\n(нет)")
    return "".join(parts)


def _clamp_rating(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def normalize_search_ai_review(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Ответ модели не является JSON-объектом")

    verdict = str(raw.get("ranking_verdict") or "").strip()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"Неверный ranking_verdict: {verdict or 'пусто'}")

    rating = _clamp_rating(raw.get("ranking_rating"))
    if rating is None:
        raise ValueError("ranking_rating должен быть 1-5")

    tags_in = raw.get("tags") or []
    if not isinstance(tags_in, list):
        tags_in = []
    tags = [t for t in (str(x).strip() for x in tags_in) if t in SEARCH_VALID_TAGS]

    improvements = raw.get("engine_improvements_ru") or []
    if isinstance(improvements, str):
        improvements = [improvements]
    if not isinstance(improvements, list):
        improvements = []
    engine_improvements_ru = [str(x).strip() for x in improvements if str(x).strip()][:10]

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

    top1 = raw.get("top1_relevant")
    if top1 is not None and not isinstance(top1, bool):
        top1 = str(top1).strip().lower() in ("1", "true", "yes")

    conf = str(raw.get("confidence") or "medium").strip().lower()
    if conf not in _VALID_CONFIDENCE:
        conf = "medium"

    suggested_step = raw.get("suggested_funnel_step")
    if suggested_step is not None and suggested_step != "":
        try:
            suggested_step = int(suggested_step)
            if suggested_step < 0 or suggested_step > 7:
                suggested_step = None
        except (TypeError, ValueError):
            suggested_step = None
    else:
        suggested_step = suggested_funnel_step_from_tags(tags)

    return {
        "ranking_verdict": verdict,
        "ranking_rating": rating,
        "tags": tags,
        "summary_ru": str(raw.get("summary_ru") or "").strip()[:2000],
        "engine_improvements_ru": engine_improvements_ru,
        "system_notes_ru": str(raw.get("system_notes_ru") or "").strip()[:2000],
        "retrieval_fix": retrieval_fix,
        "top1_relevant": top1,
        "confidence": conf,
        "suggested_funnel_step": suggested_step,
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


_SYMPTOM_RARE_TITLE_KEYWORDS = (
    "саркоид",
    "микобактер",
    "туберкул",
    "лихорадка ку",
    "орфан",
)
_SYMPTOM_COMMON_QUERY_WORDS = (
    "кашел",
    "температ",
    "лихорад",
    "озноб",
    "насморк",
    "орви",
    "простуд",
    "горл",
    "глот",
    "дисфаг",
)


def _protocol_audience_hint(path: str, title: str) -> str | None:
    try:
        import rag_server as rs

        return rs.doc_audience_hint(path, title, rs._routing or {})
    except Exception:
        blob = f"{path} {title}".lower()
        if any(x in blob for x in ("дет нас", "детск", "детс", "pediatr")):
            return "pediatric"
        if any(x in blob for x in ("взросл", "взр нас", "взр.")):
            return "adult"
        return None


def _query_audience_hint(query: str, funnel_context: dict | None = None) -> str | None:
    try:
        import rag_server as rs

        aud = rs.infer_audience_from_funnel_context(query)
        if aud:
            return aud
        fc = funnel_context or {}
        pop = str(fc.get("population") or "").strip().lower()
        if pop == "pediatric":
            return "child"
        if pop in ("adult", "pregnant", "emergency"):
            return "adult"
        return rs.infer_audience_from_query(query, rs._routing or {})
    except Exception:
        ql = (query or "").lower()
        if "детское население" in ql or "контекст подбора: детское" in ql:
            return "child"
        if "взрослое население" in ql or "контекст подбора: взрослое" in ql:
            return "adult"
        return None


def _query_has_icd_hint(query: str, icd_codes: list[str] | None) -> bool:
    if icd_codes:
        return True
    return bool(re.search(r"\b[A-TV-ZА-ЯЁ]\s*\d{2}(?:\s*[.,/\-]\s*\d{1,4})?\b", query, re.I))


def build_deterministic_search_ai_review(assist_payload: dict[str, Any]) -> dict[str, Any]:
    """Оценка выдачи без LLM — для retrieve_only и при сбое Gemini."""
    query = (assist_payload.get("query") or "").strip()
    ql = query.lower()
    protos = (assist_payload.get("llm_json") or {}).get("protocols") or []
    icd = assist_payload.get("icd_codes") or []
    retrieve_only = bool(assist_payload.get("retrieve_only"))
    funnel_context = assist_payload.get("funnel_context") if isinstance(
        assist_payload.get("funnel_context"), dict
    ) else None

    has_icd = _query_has_icd_hint(query, icd if isinstance(icd, list) else [])
    symptom_only = not has_icd

    top1 = protos[0] if protos and isinstance(protos[0], dict) else {}
    top_path = str(top1.get("path") or "")
    top_title = str(top1.get("title") or "")
    top_base = _basename(top_path).lower()

    tags: list[str] = []
    improvements: list[str] = []
    verdict = "mostly_correct"
    rating = 4
    top1_rel: bool | None = True
    retrieval_fix: dict[str, str] | None = None

    query_aud = _query_audience_hint(query, funnel_context)
    top_aud = _protocol_audience_hint(top_path, top_title) if top_path else None
    if query_aud == "adult" and top_aud == "pediatric":
        tags.append("wrong_population")
        top1_rel = False
        rating = min(rating, 2)
        verdict = "wrong"
        improvements.extend(
            [
                "При «взрослое население» в воронке отфильтровывать/опускать детские КП (дет нас, детск).",
                "Усилить doc_audience_hint для «дет нас» без подчёркивания в названии PDF.",
                "Проверить шаг 1 воронки: population=adult должен попадать в funnel_context feedback.",
            ]
        )
        if top_path:
            retrieval_fix = {
                "rejected_path": top_path,
                "chosen_path": "",
                "note": "Top-1 детский КП при контексте взрослого пациента.",
            }

    if symptom_only:
        tags.append("query_too_vague")
        improvements.append(
            "Включить обязательный шаг МКБ-10 в воронке перед списком протоколов (symptom-only)."
        )
        rating = 3
        verdict = "partially_wrong"

    common_symptoms = any(w in ql for w in _SYMPTOM_COMMON_QUERY_WORDS)
    rare_top = any(k in top_base for k in _SYMPTOM_RARE_TITLE_KEYWORDS)
    query_mentions_rare = any(k in ql for k in _SYMPTOM_RARE_TITLE_KEYWORDS)

    if symptom_only and common_symptoms and rare_top and not query_mentions_rare:
        if "wrong_protocol" not in tags:
            tags.append("wrong_protocol")
        top1_rel = False
        rating = min(rating, 2)
        verdict = "partially_wrong"
        improvements.extend(
            [
                "При symptom-only понизить вес редких КП (саркоидоз, микобактериоз, ТБ) без кода МКБ.",
                "Summary-first: boost overview-чанков с МКБ J06/J18/J20 для острых респираторных жалоб.",
                "Усилить шаг «популяция» (взрослые/дети) перед retrieval.",
            ]
        )
        if top_path:
            retrieval_fix = {
                "rejected_path": top_path,
                "chosen_path": "",
                "note": "Top-1 маловероятен для симптомного запроса без МКБ; укажите верный КП.",
            }

    if retrieve_only and not symptom_only:
        improvements.append(
            "Retrieve-only: ranking только по RAG; для аудита достаточно, для симптомов нужен шаг МКБ."
        )

    if not improvements:
        improvements = [
            "Накапливать retrieval_fix для golden set (фаза B3).",
            "Проверить Hit@3 на дашборде «Поиск · оценки».",
        ]

    summary_parts = [
        f"Детерминированная оценка ({len(protos)} протокол(ов) в выдаче).",
    ]
    if symptom_only:
        summary_parts.append(
            "Запрос без явного МКБ/диагноза — релевантность top-1 ограничена."
        )
    if top_path:
        summary_parts.append(f"Top-1: {_basename(top_path)}.")
    if top1_rel is False:
        summary_parts.append("Top-1 клинически сомнителен для формулировки запроса.")

    return {
        "ranking_verdict": verdict,
        "ranking_rating": rating,
        "tags": list(dict.fromkeys(tags)),
        "summary_ru": " ".join(summary_parts)[:2000],
        "engine_improvements_ru": improvements[:10],
        "system_notes_ru": (
            "Оценка без LLM (retrieve_only или сбой Gemini). "
            "Методист подтверждает или правит вручную."
        )[:2000],
        "retrieval_fix": retrieval_fix,
        "top1_relevant": top1_rel,
        "confidence": "low" if symptom_only else "medium",
        "suggested_funnel_step": suggested_funnel_step_from_tags(tags),
        "review_source": "deterministic_fallback",
    }


def run_methodist_search_ai_review(
    assist_payload: dict[str, Any],
    *,
    generate_fn: Callable[..., Any] | None = None,
    extract_text_fn: Callable[[Any], str] | None = None,
    parse_json_fn: Callable[[str], dict | None] | None = None,
    get_model_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Вызов LLM для предзаполнения оценки выдачи поиска."""
    if generate_fn is None or extract_text_fn is None or get_model_fn is None:
        import rag_server as rs

        generate_fn = generate_fn or rs.generate_gemini_methodist_ai_review
        extract_text_fn = extract_text_fn or rs._extract_gemini_text
        parse_json_fn = parse_json_fn or rs._try_parse_json
        get_model_fn = get_model_fn or rs.get_methodist_gemini

    try:
        model = get_model_fn()
    except Exception as exc:
        raise ValueError(f"Модель методиста недоступна: {exc!s}") from exc

    prompt = _build_prompt(assist_payload)
    try:
        resp = generate_fn(model, prompt)
    except Exception as exc:
        raise ValueError(f"Вызов модели: {exc!s}") from exc

    txt = extract_text_fn(resp)
    parsed = parse_json_fn(txt) if parse_json_fn else _try_parse_json(txt)
    if not parsed:
        raise ValueError("Модель не вернула корректный JSON для оценки поиска")

    from clinical_knowledge.gemini_model_config import methodist_gemini_model_name

    normalized = normalize_search_ai_review(parsed)
    model_name, model_warn = methodist_gemini_model_name()
    normalized["model_used"] = model_name
    if model_warn:
        normalized["model_warn"] = model_warn
    return normalized
