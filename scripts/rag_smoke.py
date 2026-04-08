#!/usr/bin/env python3
"""
Дымовой прогон RAG: POST /api/assist по фикстурам, сводка и опциональное сравнение с эталоном.

Требования: запущенный rag_server (uvicorn) и рабочий GOOGLE_API_KEY в окружении сервера.

Использование:

  # Один прогон, таблица в консоль (по умолчанию http://127.0.0.1:8000)
  python3 scripts/rag_smoke.py

  Другой хост/порт:
  RAG_SMOKE_BASE=http://127.0.0.1:9000 python3 scripts/rag_smoke.py

  Сохранить снимок ответов (пути retrieval + протоколы из JSON) в файл:
  python3 scripts/rag_smoke.py --write-baseline scripts/rag_smoke_baseline.json

  После изменений в RAG — сравнить с сохранённым эталоном (код выхода 1 при отличиях):
  python3 scripts/rag_smoke.py --compare-baseline scripts/rag_smoke_baseline.json

  Смягчить сравнение: только первые N путей retrieval и N протоколов (по умолчанию 5):
  python3 scripts/rag_smoke.py --compare-baseline scripts/rag_smoke_baseline.json --depth 3

Переменные окружения клиента:
  RAG_SMOKE_BASE   — базовый URL API (без слэша в конце)
  RAG_SMOKE_TIMEOUT — таймаут HTTP в секундах (по умолчанию 300)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "scripts" / "rag_smoke_fixtures.json"


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"Нет cases в {path}")
    return cases


def _post_assist(base: str, body: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    url = base.rstrip("/") + "/api/assist"
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, None, err_body or str(e)
    except urllib.error.URLError as e:
        return 0, None, str(e.reason if hasattr(e, "reason") else e)
    try:
        return status, json.loads(text), ""
    except json.JSONDecodeError:
        return status, None, text[:500]


def _paths_retrieval(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for r in data.get("retrieval") or []:
        if isinstance(r, dict):
            p = (r.get("path") or "").strip()
            if p:
                out.append(p)
    return out


def _paths_protocols(data: dict[str, Any]) -> list[str]:
    j = data.get("llm_json")
    if not isinstance(j, dict):
        return []
    protos = j.get("protocols")
    if not isinstance(protos, list):
        return []
    out: list[str] = []
    for pr in protos:
        if isinstance(pr, dict):
            p = (pr.get("path") or "").strip()
            if p:
                out.append(p)
    return out


def _run_case(
    base: str,
    case: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    cid = str(case.get("id") or "?")
    query = (case.get("query") or "").strip()
    if len(query) < 2:
        return {
            "id": cid,
            "ok": False,
            "error": "пустой или короткий query в фикстуре",
            "ms": 0,
            "http_status": 0,
            "retrieval_paths": [],
            "protocol_paths": [],
        }
    slugs = case.get("category_slugs")
    if not isinstance(slugs, list):
        slugs = []
    body = {"query": query, "category_slugs": slugs}
    t0 = time.perf_counter()
    status, data, err = _post_assist(base, body, timeout)
    ms = int((time.perf_counter() - t0) * 1000)
    if status != 200 or not data:
        return {
            "id": cid,
            "ok": False,
            "error": err or f"HTTP {status}",
            "ms": ms,
            "http_status": status,
            "retrieval_paths": [],
            "protocol_paths": [],
        }
    return {
        "id": cid,
        "ok": True,
        "error": "",
        "ms": ms,
        "http_status": status,
        "retrieval_paths": _paths_retrieval(data),
        "protocol_paths": _paths_protocols(data),
    }


def _truncate_paths(paths: list[str], depth: int) -> list[str]:
    if depth <= 0:
        return list(paths)
    return paths[:depth]


def _compare_one(
    a: dict[str, Any],
    b: dict[str, Any],
    depth: int,
) -> list[str]:
    diffs: list[str] = []
    aid = a.get("id")
    ra, rb = _truncate_paths(a.get("retrieval_paths") or [], depth), _truncate_paths(
        b.get("retrieval_paths") or [], depth
    )
    pa, pb = _truncate_paths(a.get("protocol_paths") or [], depth), _truncate_paths(
        b.get("protocol_paths") or [], depth
    )
    if ra != rb:
        diffs.append(f"{aid} retrieval_paths: {ra!r} != {rb!r}")
    if pa != pb:
        diffs.append(f"{aid} protocol_paths: {pa!r} != {pb!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description="Дымовой прогон POST /api/assist по фикстурам.")
    ap.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="JSON с полем cases (по умолчанию scripts/rag_smoke_fixtures.json)",
    )
    ap.add_argument(
        "--base",
        default=os.environ.get("RAG_SMOKE_BASE", "http://127.0.0.1:8000").rstrip("/"),
        help="Базовый URL сервера (или RAG_SMOKE_BASE)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("RAG_SMOKE_TIMEOUT", "300")),
        help="Таймаут запроса в секундах",
    )
    ap.add_argument(
        "--write-baseline",
        type=Path,
        metavar="FILE",
        help="Сохранить снимок результатов в JSON для последующего --compare-baseline",
    )
    ap.add_argument(
        "--compare-baseline",
        type=Path,
        metavar="FILE",
        help="Сравнить текущий прогон с сохранённым снимком (exit 1 при отличиях)",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Сколько путей retrieval/protocols учитывать при сравнении",
    )
    args = ap.parse_args()

    try:
        cases = _load_fixtures(args.fixtures)
    except OSError as e:
        print(f"Ошибка чтения фикстур: {e}", file=sys.stderr)
        return 2

    print(f"RAG smoke: base={args.base} cases={len(cases)} timeout={args.timeout}s")
    results: list[dict[str, Any]] = []
    all_ok = True
    for case in cases:
        r = _run_case(args.base, case, args.timeout)
        results.append(r)
        if not r["ok"]:
            all_ok = False
        rp = ", ".join((r.get("retrieval_paths") or [])[:3])
        pp = ", ".join((r.get("protocol_paths") or [])[:3])
        status = "OK" if r["ok"] else "FAIL"
        err = f" — {r.get('error')}" if r.get("error") else ""
        print(
            f"  [{status}] {r['id']}: {r['ms']} ms http={r.get('http_status', 0)}{err}\n"
            f"       retrieval: {rp or '—'}\n"
            f"       protocols: {pp or '—'}"
        )

    snapshot = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base,
        "fixtures": str(args.fixtures),
        "depth_note": f"Полные списки путей; при сравнении используйте --depth (сейчас {args.depth})",
        "results": results,
    }

    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Эталон записан: {args.write_baseline}")

    exit_code = 0
    if args.compare_baseline:
        if not args.compare_baseline.is_file():
            print(f"Нет файла эталона: {args.compare_baseline}", file=sys.stderr)
            return 2
        try:
            prev = json.loads(args.compare_baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Не удалось прочитать эталон: {e}", file=sys.stderr)
            return 2
        prev_results = prev.get("results")
        if not isinstance(prev_results, list):
            print("В эталоне нет results", file=sys.stderr)
            return 2
        by_id = {str(x.get("id")): x for x in prev_results if isinstance(x, dict)}
        all_diffs: list[str] = []
        for r in results:
            rid = str(r.get("id"))
            if rid not in by_id:
                all_diffs.append(f"{rid}: нет в эталоне")
                continue
            all_diffs.extend(_compare_one(by_id[rid], r, args.depth))
        if all_diffs:
            print("\nОтличия от эталона:", file=sys.stderr)
            for line in all_diffs:
                print(f"  {line}", file=sys.stderr)
            exit_code = 1
        else:
            print(f"\nСовпадает с эталоном (depth={args.depth}).")

    if not all_ok:
        return max(exit_code, 1)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
