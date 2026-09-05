#!/usr/bin/env python3
"""LLM-грейдер КЗ (MedCheckLLM-стиль), двухтировый - §6.4 ТЗ.

Первый проход - `gemini-3.6-flash` (дёшево, по всем КЗ). Спорные / низкая
уверенность / расхождение с детерминированными детекторами эскалируются на
`gemini-3.1-pro-preview` (судья). Остаток с needs_human=true - методисту.

ВАЖНО (§2.5, §9): все LLM-вызовы КЗ идут через Render (нет геоблока); сырые
ответы LLM (могут содержать ПДн) хранить только на /var/data.

Функции build_grader_prompt / parse_grader_response - чистые (unit-тестируемы
без сети). Оркестрация grade_kz_llm вызывает Gemini через gemini_lite.

CLI (на Render):
  set -a && source .env && set +a
  GEMINI_MODEL=gemini-3.6-flash python3 scripts/grade_kz_llm.py \
      --cases /var/data/.../kz_l1_2026-07_cases.jsonl --out /var/data/.../grades.jsonl \
      --limit 200 --escalate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.phi_for_llm import redact_text_for_llm  # noqa: E402

# Атомарная рубрика (§3.5) - fallback, если у протокола нет kz_checklist.
_RUBRIC_ITEMS = [
    ("A_complaints", "Жалобы задокументированы конкретно (не «жалоб нет» при наличии повода)"),
    ("A_anamnesis", "Анамнез заболевания отражает динамику/длительность"),
    ("A_objective", "Объективный статус относится к жалобам/диагнозу"),
    ("B_dx_from_data", "Диагноз логически следует из жалоб/анамнеза/осмотра"),
    ("B_icd_coded", "Диагноз кодируется по МКБ-10 и код соответствует тексту"),
    ("B_exams_adequate", "Назначенные обследования достаточны для диагноза (по протоколу)"),
    ("B_tx_matches_dx", "Лечение соответствует диагнозу и протоколу МЗ РБ"),
    ("C_red_flags", "Тревожные признаки (red flags) исключены/отработаны с маршрутизацией"),
    ("C_drug_safety", "Нет опасных сочетаний/дублей/дозовых ошибок в назначениях"),
    ("C_follow_up", "Есть план наблюдения/повторной явки/условия обращения"),
]

_CHAIN_INSTRUCTION = (
    "Проверь ЦЕПОЧКУ КЛИНИЧЕСКОЙ СОГЛАСОВАННОСТИ: жалобы+анамнез+осмотр → диагноз(+МКБ) "
    "→ обследования → лечение. Для каждого перехода определи исход: ok | gap | contradiction. "
    "Особое внимание: диагноз не следует из данных; пропущен red flag; лечение не по протоколу; "
    "опасное лекарственное сочетание; доза/мониторинг high-alert."
)

_FEWSHOT = (
    "Пример ХОРОШЕГО (кратко): жалобы на боль в горле 2 дня, осмотр - гиперемия зева, "
    "Dx J02.9 острый фарингит (код соответствует), назначен парацетамол при лихорадке, "
    "явка при ухудшении → все переходы ok.\n"
    "Пример ПЛОХОГО (кратко): жалобы на давящую загрудинную боль, Dx «ОРВИ», ЭКГ не назначена, "
    "нет маршрутизации → contradiction (диагноз игнорирует red flag ОКС), gap (нет ЭКГ)."
)

_JSON_SCHEMA = (
    '{"items":[{"id":"<id>","pass":true|false,"finding":"<что не так, кратко>",'
    '"evidence":"<цитата из КЗ>","confidence":0.0-1.0}],'
    '"chain":[{"from":"complaints","to":"diagnosis","outcome":"ok|gap|contradiction","note":"..."}],'
    '"axes":{"documentation":0-100,"clinical_concordance":0-100,"safety":0-100},'
    '"overall_pct":0-100,"verdict":"good|acceptable|review|poor|critical",'
    '"potential_harm":true|false,"confidence":0.0-1.0,"needs_human":true|false}'
)

_KZ_FIELDS = [
    ("Жалобы", "complaints"),
    ("Анамнез", "anamnesis_doctor,anamnesis_auto"),
    ("Объективный статус", "objective_status"),
    ("Данные обследований", "exam_data"),
    ("Диагноз", "clinical_diagnosis,diagnosis_main_text"),
    ("МКБ (основной)", "mkb_code_main"),
    ("Рекомендации по обследованию", "exam_recommendations"),
    ("Рекомендации по лечению", "treatment_recommendations"),
    ("Наблюдение/явка", "dispensary_info,return_date"),
]


def _txt(case: dict, keys: str) -> str:
    return " ".join(str(case.get(k) or "").strip() for k in keys.split(",") if case.get(k)).strip()


def _checklist_from_protocol(protocol_ctx: Any) -> list[tuple[str, str]]:
    if protocol_ctx is None:
        return list(_RUBRIC_ITEMS)
    kz = protocol_ctx.get("kz_checklist") if isinstance(protocol_ctx, dict) else getattr(protocol_ctx, "kz_checklist", None)
    items: list[tuple[str, str]] = []
    for i, it in enumerate(kz or []):
        if isinstance(it, str) and it.strip():
            items.append((f"P_{i}", it.strip()))
        elif isinstance(it, dict):
            t = it.get("text") or it.get("item") or it.get("name")
            if t:
                items.append((it.get("id") or f"P_{i}", str(t)))
    # рубрика всегда добавляется как базовый минимум
    return items + list(_RUBRIC_ITEMS) if items else list(_RUBRIC_ITEMS)


def build_grader_prompt(case: dict, checklist: list[tuple[str, str]], *, protocol_name: str = "") -> str:
    kz_lines = []
    for label, keys in _KZ_FIELDS:
        # Поля МИС несут шапку КЗ: ФИО, ИНП, телефон, дату рождения. Для оценки
        # по чек-листу они не нужны, а уезжают за пределы страны вместе с текстом.
        val = redact_text_for_llm(_txt(case, keys))
        kz_lines.append(f"{label}: {val or ' - '}")
    checklist_lines = "\n".join(f"- [{cid}] {txt}" for cid, txt in checklist)
    proto = f"Протокол МЗ РБ: {protocol_name}\n" if protocol_name else ""
    return (
        "Ты - клинический методист-эксперт. Оцени консультативное заключение (КЗ) врача "
        "строго по чек-листу и цепочке согласованности. Опирайся ТОЛЬКО на текст КЗ; "
        "если данных нет - это gap. Для каждого пункта дай pass/fail + цитату-обоснование.\n\n"
        f"{_CHAIN_INSTRUCTION}\n\n{_FEWSHOT}\n\n"
        f"{proto}=== ТЕКСТ КЗ ===\n" + "\n".join(kz_lines) + "\n\n"
        "=== ЧЕК-ЛИСТ (атомарные бинарные пункты) ===\n" + checklist_lines + "\n\n"
        "Верни СТРОГО JSON по схеме (без markdown):\n" + _JSON_SCHEMA
    )


def parse_grader_response(text: str) -> dict:
    """Извлечь JSON из ответа LLM (устойчиво к обрамлению markdown/мусору)."""
    if not text:
        return {"_parse_error": "empty"}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.I | re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"_parse_error": "no_json", "_raw": s[:500]}


def _should_escalate(parsed: dict, deterministic: dict | None) -> tuple[bool, str]:
    if parsed.get("_parse_error"):
        return True, "parse_error"
    conf = parsed.get("confidence")
    if isinstance(conf, (int, float)) and conf < 0.6:
        return True, "low_confidence"
    if parsed.get("needs_human"):
        return True, "needs_human"
    # расхождение с детерминированными детекторами по потенциальному вреду
    if deterministic is not None:
        det_harm = bool(deterministic.get("has_potential_harm"))
        llm_harm = bool(parsed.get("potential_harm"))
        if det_harm != llm_harm:
            return True, "harm_disagreement"
    return False, ""


def _build_model(model_name: str):
    # Через общий клиент: раньше модель создавалась здесь напрямую и без
    # safety_settings, то есть оценка КЗ шла с дефолтным порогом
    # BLOCK_MEDIUM_AND_ABOVE. На клиническом тексте (травмы, самоповреждения,
    # репродуктивное здоровье) такой порог блокирует ответ, и оценка
    # записывалась пустой без ошибки в логе.
    from clinical_knowledge.gemini_client import build_model
    from clinical_knowledge.gemini_model_config import resolve_gemini_model

    name, _warn = resolve_gemini_model(model_name)
    return build_model(name), name


def _generate_with_fallback(model_name: str, prompt: str):
    import os

    from clinical_knowledge.gemini_lite import generate_lite_json_response

    configured = [
        value.strip()
        for value in os.environ.get(
            "MO_LLM_MODEL_FALLBACKS", "gemini-3.6-flash,gemini-2.5-flash"
        ).split(",")
        if value.strip()
    ]
    candidates = list(dict.fromkeys([model_name, *configured]))
    errors: list[str] = []
    for candidate in candidates:
        try:
            model, resolved = _build_model(candidate)
            started = time.perf_counter()
            response = generate_lite_json_response(model, prompt)
            return response, resolved, int((time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}:{str(exc)[:120]}")
    raise RuntimeError("all_llm_models_failed:" + " | ".join(errors))


def grade_kz_llm(
    case: dict,
    protocol_ctx: Any = None,
    *,
    bulk_model: str = "gemini-3.6-flash",
    judge_model: str = "gemini-3.1-pro-preview",
    escalate: bool = True,
    deterministic: dict | None = None,
    protocol_name: str = "",
) -> dict:
    """Оценить КЗ грейдером. Возвращает parsed JSON + метаданные тира/эскалации."""
    from clinical_knowledge.gemini_lite import _extract_text
    from clinical_knowledge.mo_llm_usage import response_usage

    checklist = _checklist_from_protocol(protocol_ctx)
    prompt = build_grader_prompt(case, checklist, protocol_name=protocol_name)

    response, bulk_resolved, bulk_latency = _generate_with_fallback(bulk_model, prompt)
    raw = _extract_text(response)
    parsed = parse_grader_response(raw)
    prompt_tokens, completion_tokens = response_usage(response)
    calls = [{
        "tier": "bulk",
        "model": bulk_resolved,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": bulk_latency,
        "status": "parse_error" if parsed.get("_parse_error") else "ok",
    }]
    tier = "bulk"
    model_used = bulk_resolved
    esc_reason = ""

    if escalate:
        do_esc, esc_reason = _should_escalate(parsed, deterministic)
        if do_esc:
            judge_response, judge_resolved, judge_latency = _generate_with_fallback(
                judge_model, prompt
            )
            raw_j = _extract_text(judge_response)
            parsed_j = parse_grader_response(raw_j)
            judge_prompt_tokens, judge_completion_tokens = response_usage(judge_response)
            calls.append({
                "tier": "judge",
                "model": judge_resolved,
                "prompt_tokens": judge_prompt_tokens,
                "completion_tokens": judge_completion_tokens,
                "latency_ms": judge_latency,
                "status": "parse_error" if parsed_j.get("_parse_error") else "ok",
            })
            if not parsed_j.get("_parse_error"):
                parsed = parsed_j
                tier = "judge"
                model_used = judge_resolved

    parsed["_grader_tier"] = tier
    parsed["_grader_model"] = model_used
    parsed["_escalation_reason"] = esc_reason
    parsed["_llm_calls"] = calls
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True, help="kz_l1_<month>_cases.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="куда писать оценки (JSONL, /var/data)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worst-only", action="store_true", help="только КЗ с deep.has_potential_harm или низким overall")
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--bulk-model", default="gemini-3.6-flash")
    ap.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help="при --resume не считать успешными строки с _error (перепрогон после geo-block)",
    )
    ap.add_argument("--queue", type=Path, help="JSON queue containing visit_ids")
    ap.add_argument("--warehouse", type=Path, help="MO warehouse for token and cost accounting")
    ap.add_argument("--run-id", default="", help="stable pipeline run identifier")
    args = ap.parse_args()

    from clinical_knowledge.gemini_lite import gemini_available
    from clinical_knowledge.kz_deep_eval import load_drug_ctx, resolve_protocol_ctx

    if not gemini_available():
        print("GEMINI key not set - abort", file=sys.stderr)
        return 2

    done: set[str] = set()
    kept_lines: list[str] = []
    if args.resume and args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(row.get("visit_id") or "")
            if not vid:
                continue
            if args.retry_errors and (row.get("_error") or row.get("error") or row.get("status") == "error"):
                continue
            done.add(vid)
            kept_lines.append(line)
        if args.retry_errors:
            args.out.write_text(
                ("\n".join(kept_lines) + ("\n" if kept_lines else "")),
                encoding="utf-8",
            )

    cases = []
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if args.worst_only:
        cases = [c for c in cases if (c.get("deep") or {}).get("has_potential_harm")
                 or (c.get("overall_pct") or 100) < 60]
    if args.queue and args.queue.is_file():
        queued = {
            str(value)
            for value in (json.loads(args.queue.read_text(encoding="utf-8")).get("visit_ids") or [])
        }
        cases = [case for case in cases if str(case.get("visit_id") or "") in queued]
    if args.limit:
        cases = cases[: args.limit]

    _ = load_drug_ctx()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with args.out.open("a", encoding="utf-8") as fout:
        for c in cases:
            vid = str(c.get("visit_id") or "")
            if args.resume and vid in done:
                continue
            proto = resolve_protocol_ctx(c)
            try:
                res = grade_kz_llm(
                    c, protocol_ctx=proto, escalate=args.escalate,
                    deterministic=c.get("deep"),
                    bulk_model=args.bulk_model, judge_model=args.judge_model,
                    protocol_name=(proto or {}).get("name") or "",
                )
                res["visit_id"] = vid
                if args.warehouse:
                    from clinical_knowledge.mo_llm_usage import record_llm_usage

                    visit_day = str(
                        c.get("visit_date") or c.get("date") or ""
                    ).strip()[:10] or None
                    for call in res.get("_llm_calls") or []:
                        record_llm_usage(
                            args.warehouse,
                            run_id=args.run_id or f"llm-{args.out.stem}",
                            case_id=vid,
                            usage_date=visit_day,
                            **call,
                        )
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                fout.write(json.dumps({"visit_id": vid, "_error": str(e)[:200]}, ensure_ascii=False) + "\n")
                fout.flush()
            if (ok + fail) % 25 == 0:
                print(f"graded {ok+fail} (ok={ok} fail={fail})", flush=True)
    print(f"DONE graded ok={ok} fail={fail} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
