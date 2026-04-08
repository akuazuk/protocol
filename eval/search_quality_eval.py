#!/usr/bin/env python3
"""
Оценка качества поиска протоколов: полный retrieve() как в проде, включая опциональный
Gemini embed-rerank (если заданы RAG_GEMINI_EMBED_RERANK=1 и GOOGLE_API_KEY).

Для каждого кейса:
  — метрики (пустой отбор, совпадение must_substrings, ожидаемые пути);
  — эвристический «план улучшений» (что проверить в коде и данных);
  — при --gemini-advice — краткий анализ и шаги от модели (нужен API-ключ).

Примеры:

  python3 eval/search_quality_eval.py --mini --golden eval/golden_queries.jsonl

  # как в проде: семантический rerank (нужен ключ в окружении)
  export GOOGLE_API_KEY=...
  python3 eval/search_quality_eval.py --embed-on --golden eval/golden_queries.jsonl

  python3 eval/search_quality_eval.py --query "острый бронхит кашель" --gemini-advice --report-json report.json

Формат строки в JSONL (все поля кроме query необязательны):

  query                    — текст запроса
  must_substrings          — подстроки, которые должны быть в объединённом тексте топ-чанков
  expected_any_path_contains — список фрагментов пути к PDF (хотя бы один из топа)
  forbidden_substrings     — не должны встречаться в объединённом тексте топа
  min_chunks               — минимальное число строк в выдаче (провал, если меньше)
  expect_empty             — true: успех только при пустой выдаче (негативный кейс)
  notes                    — комментарий для человека
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_rs = str(ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)

from eval.retrieval_checks import (
    check_expected_paths,
    check_forbidden_substrings,
    check_must_substrings,
    score_metrics,
    validate_retrieval_schema,
)


def _ensure_repo_path() -> None:
    s = str(ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def apply_mini_fixture_env() -> None:
    p = ROOT / "tests" / "fixtures" / "chunks.mini.jsonl"
    if not p.is_file():
        raise SystemExit(f"Нет файла фикстуры: {p}")
    os.environ.setdefault("RAG_CHUNKS_JSONL", str(p))
    os.environ.setdefault("RAG_CHUNKS_SOURCE", "jsonl")


def load_golden_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


@dataclass
class CaseReport:
    index: int
    query: str
    notes: str
    ok: bool
    retrieval_empty: bool
    chunks: int
    unique_paths: int
    embed_rerank_used: bool
    must_ok: bool | None
    must_missing: list[str]
    path_ok: bool | None
    path_detail: str | None
    schema_warnings: list[str] = field(default_factory=list)
    expect_empty: bool = False
    expect_empty_ok: bool | None = None
    min_chunks: int | None = None
    min_chunks_ok: bool | None = None
    forbidden_ok: bool | None = None
    forbidden_found: list[str] = field(default_factory=list)
    top_score: float | None = None
    score_spread: float | None = None
    top_path_preview: str = ""
    heuristic_issues: list[str] = field(default_factory=list)
    heuristic_plan: list[str] = field(default_factory=list)
    improvement_hints: list[str] = field(default_factory=list)
    gemini_analysis: str | None = None
    gemini_plan: list[str] = field(default_factory=list)


def build_heuristic_plan(
    rep: CaseReport,
    *,
    api_key_present: bool,
    embed_requested: bool,
) -> None:
    issues: list[str] = []
    plan: list[str] = []
    hints: list[str] = []

    if rep.expect_empty and rep.expect_empty_ok is False:
        issues.append("Ожидалась пустая выдача, но retrieve вернул чанки.")
        plan.append(
            "Ужесточить фильтры: RAG_ROUTING, штрафы в symptom_routing, менее общие токены в запросе."
        )
        plan.append("Проверить RAG_GENERIC_LEX_* — не проходят ли слишком широкие совпадения.")

    if rep.min_chunks_ok is False:
        issues.append(
            f"Мало чанков в выдаче (ожидалось min_chunks, получено {rep.chunks})."
        )
        plan.append("Поднять RAG_MAX_CHUNKS / RAG_EMBED_POOL или снизить пороги отсечения в retrieve.")
        plan.append("Проверить symptom_routing — не занижен ли скор по возрасту/рубрике.")

    if rep.forbidden_ok is False:
        issues.append(f"В выдаче есть запрещённые подстроки: {rep.forbidden_found}")
        plan.append("Скорректировать бусты рубрик/МКБ или разметку протоколов в meta.")
        plan.append("Рассмотреть негативные примеры в golden и отдельный регресс на «анти-паттерны».")

    if rep.retrieval_empty and not rep.expect_empty:
        issues.append("Пустой отбор: лексика/BM25 не нашли ни одного чанка.")
        plan.append(
            "Проверить корпус (RAG_CHUNKS_*): есть ли протоколы по теме и как разбиты чанки."
        )
        plan.append(
            "Ослабить запрос (синонимы, код МКБ), проверить tokenize_ru и RAG_GENERIC_LEX_*."
        )
        plan.append(
            "Просмотреть symptom_routing.json: нет ли штрафа возраста/рубрики для релевантных чанков."
        )
        if embed_requested and api_key_present:
            plan.append(
                "При пустом пуле до rerank семантика не поможет — сначала лексический проход."
            )

    if rep.must_ok is False:
        issues.append("В топ-чанках нет обязательных подстрок (контент не совпадает с ожиданием).")
        plan.append(
            "Пересобрать чанки (chunk_build / pipeline): title, lex_text, границы абзацев."
        )
        plan.append(
            "Проверить, не отрезан ли нужный фрагмент в excerpt (RAG_EXCERPT_CHARS)."
        )

    if rep.path_ok is False:
        issues.append("В топе нет ожидаемого файла протокола (путь не совпадает с эталоном).")
        plan.append(
            "Настроить усиление рубрики/МКБ: RAG_CATEGORY_BOOST_*, icd_codes в запросе, passport/meta протокола."
        )
        plan.append(
            "Проверить symptom_routing.json и веса lex vs BM25 (RAG_LEX_BM25_ALPHA, RAG_EMBED_POOL)."
        )
        plan.append(
            "Включить или отладить Gemini embed-rerank (RAG_GEMINI_EMBED_RERANK=1), если ключ есть."
        )

    if not embed_requested and api_key_present:
        if not (rep.expect_empty and rep.retrieval_empty):
            issues.append(
                "Семантический rerank выключен — порядок чанков может отличаться от «боевого» режима."
            )
            plan.append(
                "Запускать eval с --embed-on при наличии GOOGLE_API_KEY для сопоставимости с продом."
            )

    if embed_requested and not rep.embed_rerank_used and api_key_present:
        issues.append("Embed-rerank не применился (ошибка API или пустой пул до rerank).")
        plan.append("Смотреть _retrieval_embed_meta / логи; проверить GEMINI_EMBEDDING_MODEL и квоты.")

    if rep.schema_warnings:
        issues.append("Схема результатов retrieve: " + "; ".join(rep.schema_warnings[:4]))
        plan.append("Синхронизировать ключи path, excerpt, score с клиентом и тестами.")

    if rep.chunks > 1 and rep.unique_paths == 1:
        hints.append(
            "Все чанки с одного PDF — ожидаемо при max_per_path>1; иначе расширьте корпус по теме."
        )
    if (
        rep.top_score is not None
        and rep.score_spread is not None
        and rep.chunks >= 2
        and rep.score_spread < 0.02
    ):
        hints.append(
            f"Малый разрыв скоров (spread={rep.score_spread}): порядок нестабилен без семантики — embed-rerank."
        )
    if rep.top_score is not None and rep.top_score < 0.12 and not rep.retrieval_empty:
        hints.append(
            f"Низкий top_score ({rep.top_score}) — слабое совпадение; уточнить запрос или BM25/лексику."
        )

    rep.heuristic_issues = issues
    rep.heuristic_plan = plan
    rep.improvement_hints = hints


_GEMINI_ADVICE_PROMPT = """Ты помощник по улучшению RAG-поиска по медицинским протоколам (русский язык).
Дан запрос пользователя и фрагменты из топа выдачи (path + краткий excerpt).
Ответь СТРОГО одним JSON-объектом без markdown:
{{"relevance": <число 1-5 насколько топ релевантен запросу>, "main_issue": "<одна короткая формулировка или пустая строка>", "steps": ["шаг 1", "шаг 2", "до 5 пунктов — конкретные действия для разработчика/контента"]}}

Запрос:
{query}

Топ выдачи (сокращено):
{blocks}

Если выдача пустая, relevance=1, опиши в steps что добавить в корпус или как переформулировать запрос.
"""


def gemini_advice_for_case(
    query: str,
    retrieved: list[dict],
    *,
    model_name: str | None = None,
) -> tuple[str | None, list[str]]:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, []
    try:
        import google.generativeai as genai
    except ImportError:
        return None, ["Установите google-generativeai для --gemini-advice"]

    genai.configure(api_key=key)
    name = (model_name or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    model = genai.GenerativeModel(name)

    blocks: list[str] = []
    for i, r in enumerate(retrieved[:6], 1):
        p = r.get("path") or ""
        ex = (r.get("excerpt") or "")[:400].replace("\n", " ")
        blocks.append(f"[{i}] {p}\n{ex}")
    if not blocks:
        blocks.append("(пустая выдача)")

    prompt = _GEMINI_ADVICE_PROMPT.format(
        query=query,
        blocks="\n---\n".join(blocks),
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024},
        )
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        return None, [f"Gemini: {e!s}"]

    m = re.search(r"\{[\s\S]*\}", text)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return text, []

    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    rel = data.get("relevance")
    issue = data.get("main_issue") or ""
    summary_parts: list[str] = []
    if rel is not None:
        summary_parts.append(f"relevance={rel}")
    if issue:
        summary_parts.append(str(issue))
    summary = "; ".join(summary_parts) if summary_parts else text[:500]
    out_steps = [str(s) for s in steps if s][:8]
    return summary, out_steps


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def evaluate_one(
    idx: int,
    case: dict[str, Any],
    retrieve,
    *,
    max_chunks: int,
    max_per_path: int,
    gemini_advice: bool,
    api_key_present: bool,
    embed_requested: bool,
) -> CaseReport:
    q = (case.get("query") or "").strip()
    notes = str(case.get("notes") or "").strip()
    must = case.get("must_substrings") or []
    if not isinstance(must, list):
        must = []
    path_fr = case.get("expected_any_path_contains") or case.get("expected_path_fragments")
    if path_fr is None:
        path_fr = []
    if not isinstance(path_fr, list):
        path_fr = []
    forbidden = case.get("forbidden_substrings") or []
    if not isinstance(forbidden, list):
        forbidden = []
    expect_empty = _truthy(case.get("expect_empty"))
    min_chunks_raw = case.get("min_chunks")
    min_chunks: int | None = None
    if min_chunks_raw is not None:
        try:
            min_chunks = max(0, int(min_chunks_raw))
        except (TypeError, ValueError):
            min_chunks = None

    import rag_server as _rs  # noqa: E402

    out = retrieve(q, max_chunks=max_chunks, max_per_path=max_per_path) if q else []

    meta = getattr(_rs, "_retrieval_embed_meta", None)
    embed_used = bool(meta and meta.get("used"))

    empty = len(out) == 0
    schema_warnings = validate_retrieval_schema(out)
    sm = score_metrics(out)
    top_sc = sm.get("top_score")
    spread = sm.get("spread")
    top_path_preview = ""
    if out and isinstance(out[0], dict):
        top_path_preview = str(out[0].get("path") or "")[:96]

    must_ok: bool | None = None
    must_miss: list[str] = []
    if must:
        must_ok, must_miss = check_must_substrings(out, must)
    path_ok: bool | None = None
    path_detail: str | None = None
    if path_fr:
        path_ok, path_detail = check_expected_paths(out, path_fr)

    forbidden_ok: bool | None = None
    forbidden_found: list[str] = []
    if forbidden:
        forbidden_ok, forbidden_found = check_forbidden_substrings(out, forbidden)

    min_chunks_ok: bool | None = None
    if min_chunks is not None:
        min_chunks_ok = len(out) >= min_chunks

    expect_empty_ok: bool | None = None
    if expect_empty:
        expect_empty_ok = empty

    ok = True
    if expect_empty:
        ok = bool(expect_empty_ok)
    else:
        if must and must_ok is False:
            ok = False
        if path_fr and path_ok is False:
            ok = False
        if empty and (must or path_fr):
            ok = False
        if forbidden and forbidden_ok is False:
            ok = False
        if min_chunks_ok is False:
            ok = False

    rep = CaseReport(
        index=idx,
        query=q,
        notes=notes,
        ok=ok,
        retrieval_empty=empty,
        chunks=len(out),
        unique_paths=len({str(r.get("path")) for r in out if isinstance(r, dict)}),
        embed_rerank_used=embed_used,
        must_ok=must_ok,
        must_missing=must_miss,
        path_ok=path_ok,
        path_detail=path_detail,
        schema_warnings=schema_warnings,
        expect_empty=expect_empty,
        expect_empty_ok=expect_empty_ok,
        min_chunks=min_chunks,
        min_chunks_ok=min_chunks_ok,
        forbidden_ok=forbidden_ok,
        forbidden_found=forbidden_found,
        top_score=float(top_sc) if top_sc is not None else None,
        score_spread=float(spread) if spread is not None else None,
        top_path_preview=top_path_preview,
    )
    build_heuristic_plan(rep, api_key_present=api_key_present, embed_requested=embed_requested)

    if gemini_advice and q:
        gsum, gsteps = gemini_advice_for_case(q, out)
        rep.gemini_analysis = gsum
        rep.gemini_plan = gsteps

    return rep


def report_to_dict(r: CaseReport) -> dict[str, Any]:
    d = asdict(r)
    return d


def summarize_failure_criteria(reports: list[CaseReport]) -> None:
    bad = [r for r in reports if not r.ok]
    if not bad:
        return
    ctr: Counter[str] = Counter()
    for r in bad:
        if r.must_ok is False:
            ctr["must_substrings"] += 1
        if r.path_ok is False:
            ctr["expected_path"] += 1
        if r.forbidden_ok is False:
            ctr["forbidden_substrings"] += 1
        if r.min_chunks_ok is False:
            ctr["min_chunks"] += 1
        if r.expect_empty_ok is False:
            ctr["expect_empty"] += 1
    if ctr:
        parts = [f"{k}={v}" for k, v in sorted(ctr.items())]
        print(f"Сводка провалов по критериям: {', '.join(parts)}")


def print_case_human(r: CaseReport) -> None:
    status = "OK" if r.ok else "FAIL"
    print(f"\n{'═' * 72}")
    print(
        f"#{r.index}  [{status}]  chunks={r.chunks}  paths={r.unique_paths}  embed_rerank={r.embed_rerank_used}"
    )
    print(f"query: {r.query!r}")
    if r.notes:
        print(f"notes: {r.notes}")
    if r.top_path_preview:
        print(f"top_path: {r.top_path_preview!r}")
    if r.top_score is not None:
        extra = f"  spread={r.score_spread}" if r.score_spread is not None else ""
        print(f"scores: top={r.top_score}{extra}")
    if r.schema_warnings:
        print(f"schema: {r.schema_warnings}")
    if r.expect_empty:
        print(f"expect_empty: {'да' if r.expect_empty_ok else 'нет — непустая выдача'}")
    if r.min_chunks is not None:
        mc_ok = r.min_chunks_ok
        print(f"min_chunks (>={r.min_chunks}): {'да' if mc_ok else 'нет'}")
    if r.must_ok is not None:
        print(f"must_substrings: {'да' if r.must_ok else 'нет — ' + str(r.must_missing)}")
    if r.path_ok is not None:
        print(f"expected path fragments: {'да' if r.path_ok else 'нет — ' + (r.path_detail or '')}")
    if r.forbidden_ok is not None:
        print(
            f"forbidden_substrings: {'нет совпадений' if r.forbidden_ok else 'найдено: ' + str(r.forbidden_found)}"
        )
    if r.heuristic_issues:
        print("Диагностика:")
        for x in r.heuristic_issues:
            print(f"  • {x}")
    if r.heuristic_plan:
        print("План (эвристика):")
        for i, x in enumerate(r.heuristic_plan, 1):
            print(f"  {i}. {x}")
    if r.improvement_hints:
        print("Доп. подсказки:")
        for x in r.improvement_hints:
            print(f"  · {x}")
    if r.gemini_analysis or r.gemini_plan:
        print("Анализ Gemini:")
        if r.gemini_analysis:
            print(f"  {r.gemini_analysis}")
        for i, x in enumerate(r.gemini_plan, 1):
            print(f"  {i}. {x}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Оценка качества поиска: полный retrieve + диагностика + опционально Gemini"
    )
    ap.add_argument("--mini", action="store_true", help="фикстурный мини-корпус tests/fixtures")
    ap.add_argument("--query", "-q", type=str, default="", help="один запрос")
    ap.add_argument("--golden", type=Path, default=None, help="JSONL с кейсами")
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--max-per-path", type=int, default=2)
    ap.add_argument(
        "--embed-on",
        action="store_true",
        help="включить RAG_GEMINI_EMBED_RERANK=1 на время запуска (нужен GOOGLE_API_KEY)",
    )
    ap.add_argument("--embed-off", action="store_true", help="принудительно выключить embed-rerank")
    ap.add_argument(
        "--gemini-advice",
        action="store_true",
        help="запросить у Gemini краткий анализ и шаги (по каждому кейсу / одному запросу)",
    )
    ap.add_argument("--report-json", type=Path, default=None, help="сохранить полный отчёт JSON")
    args = ap.parse_args()

    if args.embed_on and args.embed_off:
        raise SystemExit("Нельзя одновременно --embed-on и --embed-off")

    if args.mini:
        apply_mini_fixture_env()

    if args.embed_on:
        os.environ["RAG_GEMINI_EMBED_RERANK"] = "1"
    elif args.embed_off:
        os.environ["RAG_GEMINI_EMBED_RERANK"] = "0"
    else:
        os.environ.setdefault("RAG_GEMINI_EMBED_RERANK", "1")

    _ensure_repo_path()
    from rag_server import retrieve  # noqa: E402

    api_key_present = bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    embed_requested = os.environ.get("RAG_GEMINI_EMBED_RERANK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    reports: list[CaseReport] = []

    if args.golden is not None:
        path = args.golden if args.golden.is_absolute() else ROOT / args.golden
        if not path.is_file():
            raise SystemExit(f"Файл не найден: {path}")
        cases = load_golden_lines(path)
        for j, case in enumerate(cases, 1):
            rep = evaluate_one(
                j,
                case,
                retrieve,
                max_chunks=args.max_chunks,
                max_per_path=args.max_per_path,
                gemini_advice=args.gemini_advice,
                api_key_present=api_key_present,
                embed_requested=embed_requested,
            )
            reports.append(rep)
            print_case_human(rep)
        failed = sum(1 for r in reports if not r.ok)
        print(f"\nИтого: {len(reports)} кейсов, провалов: {failed}")
        summarize_failure_criteria(reports)
    else:
        q = (args.query or "").strip()
        if not q:
            ap.print_help()
            print("\nУкажите --query или --golden", file=sys.stderr)
            return 2
        rep = evaluate_one(
            1,
            {"query": q},
            retrieve,
            max_chunks=args.max_chunks,
            max_per_path=args.max_per_path,
            gemini_advice=args.gemini_advice,
            api_key_present=api_key_present,
            embed_requested=embed_requested,
        )
        reports.append(rep)
        print_case_human(rep)

    if args.report_json:
        payload = {
            "embed_rerank_env": os.environ.get("RAG_GEMINI_EMBED_RERANK"),
            "api_key_present": api_key_present,
            "cases": [report_to_dict(r) for r in reports],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if str(args.report_json) == "-":
            print(text)
        else:
            args.report_json.write_text(text, encoding="utf-8")
            print(f"\nОтчёт записан: {args.report_json}")

    return 1 if any(not r.ok for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
