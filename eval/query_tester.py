#!/usr/bin/env python3
"""
Тестировщик запросов и поиска протоколов (лексический retrieve без вызова /api/assist и без Gemini).

Использование:

  # Один запрос, таблица путей и отрывков (нужен настроенный корпус в RAG_*)
  python3 eval/query_tester.py --query "кашель бронхит J20"

  # Тот же режим на минимальной фикстуре (как в pytest)
  python3 eval/query_tester.py --mini --query "кашель бронхит J20"

  # Прогон golden (те же правила, что eval/search_quality_eval.py: must, пути, min_chunks, forbidden, expect_empty)
  python3 eval/query_tester.py --mini --golden eval/golden_queries.jsonl

Переменные окружения те же, что у rag_server (RAG_CHUNKS_JSONL, RAG_CHUNKS_SOURCE, …).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_rs = str(ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)

def apply_mini_fixture_env() -> None:
    """Мини-корпус из tests/fixtures — как tests/conftest.py."""
    p = ROOT / "tests" / "fixtures" / "chunks.mini.jsonl"
    if not p.is_file():
        raise SystemExit(f"Нет файла фикстуры: {p}")
    os.environ.setdefault("RAG_CHUNKS_JSONL", str(p))
    os.environ.setdefault("RAG_CHUNKS_SOURCE", "jsonl")
    os.environ.setdefault("RAG_GEMINI_EMBED_RERANK", "0")


def load_golden_lines(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def print_retrieval_table(retrieved: list[dict]) -> None:
    if not retrieved:
        print("(пустой отбор — уточните запрос или проверьте корпус RAG_CHUNKS_*)")
        return
    w = os.get_terminal_size().columns if sys.stdout.isatty() else 100
    sep = "─" * min(w, 80)
    for i, r in enumerate(retrieved, 1):
        p = r.get("path") or ""
        sc = r.get("score")
        cat = r.get("category") or ""
        ex = (r.get("excerpt") or "").replace("\n", " ")[:500]
        print(sep)
        print(f"[{i}] score={sc}  category={cat}")
        print(f"    path: {p}")
        print(f"    excerpt: {ex}{'…' if len(str(r.get('excerpt') or '')) > 500 else ''}")
    print(sep)


def run_golden(
    golden_path: Path,
    retrieve,
    *,
    max_chunks: int,
    max_per_path: int,
    verbose: bool,
) -> int:
    from eval.search_quality_eval import evaluate_one  # noqa: E402

    api_key_present = bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    embed_requested = os.environ.get("RAG_GEMINI_EMBED_RERANK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    cases = load_golden_lines(golden_path)
    failed = 0
    for j, case in enumerate(cases, 1):
        q = (case.get("query") or "").strip()
        if not q:
            print(f"#{j} SKIP: пустой query")
            continue
        rep = evaluate_one(
            j,
            case,
            retrieve,
            max_chunks=max_chunks,
            max_per_path=max_per_path,
            gemini_advice=False,
            api_key_present=api_key_present,
            embed_requested=embed_requested,
        )
        status = "OK" if rep.ok else "FAIL"
        print(
            f"#{j} [{status}] query={q!r} chunks={rep.chunks} paths={rep.unique_paths}"
        )
        if rep.notes:
            print(f"    notes: {rep.notes}")
        if not rep.ok and verbose:
            for x in rep.heuristic_issues[:6]:
                print(f"    ! {x}")
            for i, x in enumerate(rep.heuristic_plan[:5], 1):
                print(f"    {i}. {x}")
        elif not rep.ok:
            if rep.must_missing:
                print(f"    missing must: {rep.must_missing}")
            if rep.path_detail and rep.path_ok is False:
                print(f"    path: {rep.path_detail}")
            if rep.forbidden_found:
                print(f"    forbidden hit: {rep.forbidden_found}")
            if rep.min_chunks_ok is False:
                print(f"    min_chunks: ожидалось ≥{rep.min_chunks}, получено {rep.chunks}")
            if rep.expect_empty_ok is False:
                print("    expect_empty: получена непустая выдача")
        if not rep.ok:
            failed += 1
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка retrieve() и golden-запросов")
    ap.add_argument("--mini", action="store_true", help="использовать tests/fixtures/chunks.mini.jsonl")
    ap.add_argument("--query", "-q", type=str, default="", help="один запрос")
    ap.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="путь к JSONL (формат eval/golden_queries.jsonl)",
    )
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--max-per-path", type=int, default=2)
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="при --golden: подробности провала (диагностика и план)",
    )
    ap.add_argument("--json", action="store_true", help="вывод одного запроса в JSON (stdout)")
    args = ap.parse_args()

    if args.mini:
        apply_mini_fixture_env()

    # Запуск как `python eval/query_tester.py` — корень репо должен быть в sys.path
    root_s = str(ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    # Импорт после выставления RAG_* (как в conftest)
    from rag_server import retrieve  # noqa: E402

    if args.golden is not None:
        path = args.golden if args.golden.is_absolute() else ROOT / args.golden
        if not path.is_file():
            raise SystemExit(f"Файл не найден: {path}")
        return run_golden(
            path,
            retrieve,
            max_chunks=args.max_chunks,
            max_per_path=args.max_per_path,
            verbose=args.verbose,
        )

    q = (args.query or "").strip()
    if not q:
        ap.print_help()
        print("\nУкажите --query \"...\" или --golden файл.jsonl", file=sys.stderr)
        return 2

    out = retrieve(q, max_chunks=args.max_chunks, max_per_path=args.max_per_path)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out else 1

    print_retrieval_table(out)
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
