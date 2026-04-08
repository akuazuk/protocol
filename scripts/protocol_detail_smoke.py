#!/usr/bin/env python3
"""
Дымовой вызов POST /api/protocol-detail (развёрнутая выдержка по протоколу).

Требуется: uvicorn с rag_server, GOOGLE_API_KEY, собранные чанки.

  python3 scripts/protocol_detail_smoke.py

  RAG_SMOKE_BASE=http://127.0.0.1:8000 python3 scripts/protocol_detail_smoke.py --fixtures scripts/protocol_detail_smoke_fixtures.json

В protocol_detail_smoke_fixtures.json поле path можно оставить пустым — возьмётся первый path из protocols.json.
Переменные: RAG_SMOKE_BASE, RAG_SMOKE_TIMEOUT (сек., по умолчанию 300).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIX = ROOT / "scripts" / "protocol_detail_smoke_fixtures.json"


def _first_protocol_path() -> str:
    p = ROOT / "protocols.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(data, list) and data:
        path = (data[0].get("path") or "").strip()
        return path
    return ""


def _post(base: str, body: dict, timeout: float) -> tuple[int, dict | None, str]:
    url = base.rstrip("/") + "/api/protocol-detail"
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), json.loads(text), ""
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            j = json.loads(err)
            return e.code, j, err
        except json.JSONDecodeError:
            return e.code, None, err
    except urllib.error.URLError as e:
        return 0, None, str(e.reason if hasattr(e, "reason") else e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Дымовой POST /api/protocol-detail")
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIX)
    ap.add_argument(
        "--base",
        default=os.environ.get("RAG_SMOKE_BASE", "http://127.0.0.1:8000").rstrip("/"),
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("RAG_SMOKE_TIMEOUT", "300")),
    )
    args = ap.parse_args()
    data = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if not cases:
        print("Нет cases", file=sys.stderr)
        return 2
    fallback_path = _first_protocol_path()
    print(f"protocol-detail smoke: base={args.base} cases={len(cases)}")
    ok_all = True
    for c in cases:
        cid = c.get("id") or "?"
        path = (c.get("path") or "").strip() or fallback_path
        if not path:
            print(f"  [{cid}] SKIP: нет path и пустой protocols.json", file=sys.stderr)
            ok_all = False
            continue
        body = {
            "query": (c.get("query") or "=== Жалобы и вопрос ===\nТест.").strip(),
            "path": path,
            "title": (c.get("title") or "")[:2000],
            "protocol_confidence": 0.85,
        }
        if c.get("extract_focus"):
            body["extract_focus"] = str(c.get("extract_focus"))[:32]
        if c.get("client_rag_support") is not None:
            try:
                body["client_rag_support"] = float(c.get("client_rag_support"))
            except (TypeError, ValueError):
                pass
        t0 = time.perf_counter()
        status, j, err = _post(args.base, body, args.timeout)
        ms = int((time.perf_counter() - t0) * 1000)
        if status != 200 or not j:
            print(f"  FAIL {cid} http={status} {ms}ms {err[:200]}")
            ok_all = False
            continue
        cd = j.get("clinical_detail") or {}
        ext = (cd.get("extraction") or {}) if isinstance(cd, dict) else {}
        inv = len(ext.get("investigations") or []) if isinstance(ext.get("investigations"), list) else 0
        med = len(ext.get("medications") or []) if isinstance(ext.get("medications"), list) else 0
        tm = len(ext.get("treatment_methods") or []) if isinstance(ext.get("treatment_methods"), list) else 0
        mf = bool((ext.get("monitoring_frequency") or "").strip())
        trunc = cd.get("detail_prompt_truncated")
        low = cd.get("low_protocol_match_support")
        print(
            f"  OK {cid} {ms}ms path={path[:56]}… "
            f"inv={inv} med={med} tm={tm} mf={'yes' if mf else 'no'} "
            f"trunc={trunc} low_sup={low}"
        )
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
