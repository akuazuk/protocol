#!/usr/bin/env python3
"""Batch LLM-судья серой зоны Dx↔МКБ (только GCE / night).

Не запускать с Mac для живого Gemini (правило gemini-via-render).
Default off: нужен MO_ICD_LLM_REVIEW=1.

  MO_ICD_LLM_REVIEW=1 PYTHONPATH=. python3 scripts/run_mo_icd_llm_review.py \\
    --date 2026-08-04 --medical-exams-root /var/data/medical_exams \\
    --limit 20 --dry-run

Out: $DATA/llm_icd_review/Y/M/D/reviews.jsonl (не коммитить PHI).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_icd_llm_review import (  # noqa: E402
    ENGINE,
    icd_llm_review_enabled,
    review_one,
)
from clinical_knowledge.mo_icd_match_pipeline import evaluate_mo_icd_match  # noqa: E402

MINSK = ZoneInfo("Europe/Minsk")

_CLINICAL_KEYS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "clinical_diagnosis",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "manipulations",
    "mis_diagnos",
    "mkb_code_main",
    "mkb_code_agreement",
    "mkb_code_mis",
)


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_date(raw: str) -> str:
    text = (raw or "").strip().lower()
    today = datetime.now(MINSK).date()
    if text in {"yesterday", "вчера"}:
        return (today - timedelta(days=1)).isoformat()
    if text in {"today", "сегодня"}:
        return today.isoformat()
    date.fromisoformat(text)
    return text


def _case_from_csv(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _CLINICAL_KEYS:
        val = row.get(key)
        if val not in (None, ""):
            out[key] = val
    for key in ("visit_id", "mis_id", "patient_id"):
        if row.get(key):
            out[key] = row[key]
    return out


def load_day_cases(data_root: Path, day: str) -> list[dict[str, Any]]:
    d = date.fromisoformat(day)
    secure = data_root / "secure_cases" / f"{d:%Y}" / f"{d:%m}"
    csv_path = secure / f"mo_{day}.csv"
    cases: list[dict[str, Any]] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cases.append(_case_from_csv(row))
    return cases


def _generate_gemini(prompt: str, *, model_name: str) -> tuple[str, int]:
    from clinical_knowledge.gemini_lite import get_lite_gemini_model
    from clinical_knowledge.gemini_model_config import resolve_gemini_model

    resolved, _warn = resolve_gemini_model(model_name)
    prev = os.environ.get("GEMINI_MODEL")
    os.environ["GEMINI_MODEL"] = resolved
    try:
        model = get_lite_gemini_model()
        if model is None:
            raise RuntimeError("Gemini model unavailable (нет ключа?)")
        t0 = time.perf_counter()
        from clinical_knowledge import gemini_lite

        if hasattr(gemini_lite, "generate_text"):
            text = gemini_lite.generate_text(prompt, max_out=1024)
        else:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or ""
        ms = int((time.perf_counter() - t0) * 1000)
        return text, ms
    finally:
        if prev is None:
            os.environ.pop("GEMINI_MODEL", None)
        else:
            os.environ["GEMINI_MODEL"] = prev


def judge_case(
    case: dict[str, Any],
    *,
    model_name: str,
    dry_run: bool,
    generate_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    vid = str(case.get("visit_id") or case.get("mis_id") or "").strip()
    pipe = evaluate_mo_icd_match(case)
    base: dict[str, Any] = {
        "visit_id": vid,
        "mis_id": str(case.get("mis_id") or ""),
        "engine": ENGINE,
        "needs_llm_review": bool(pipe.get("needs_llm_review")),
        "pipeline_verdict": pipe.get("pipeline_verdict"),
        "chip_status": (pipe.get("chip") or {}).get("status"),
    }
    if not pipe.get("needs_llm_review"):
        base["skipped"] = True
        base["reason"] = "not_needed"
        return base
    if dry_run:
        from clinical_knowledge.mo_icd_llm_review import build_llm_review_pack, build_prompt

        pack = build_llm_review_pack(pipe)
        base.update(
            {
                "skipped": True,
                "reason": "dry_run",
                "pack": pack,
                "prompt_chars": len(build_prompt(pack)) if pack else 0,
            }
        )
        return base

    def _gen(prompt: str) -> str:
        if generate_fn is not None:
            return generate_fn(prompt)
        text, ms = _generate_gemini(prompt, model_name=model_name)
        base["latency_ms"] = ms
        return text

    # temporarily enable for review_one if caller set env
    result = review_one(pipe, generate_fn=_gen)
    base.update(
        {
            "skipped": result.get("skipped"),
            "reason": result.get("reason"),
            "review": result.get("review"),
            "findings": result.get("findings") or [],
            # pack без полного diag в лог-сводке - уже clipped в pack
            "pack": {
                "code": (result.get("pack") or {}).get("code"),
                "candidates_n": len((result.get("pack") or {}).get("candidates") or []),
            }
            if result.get("pack")
            else None,
        }
    )
    return base


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD|yesterday")
    ap.add_argument(
        "--medical-exams-root",
        type=Path,
        default=Path(os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams"),
    )
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--model",
        default=os.environ.get("MO_ICD_LLM_REVIEW_MODEL")
        or os.environ.get("MO_LLM_ACTION_JUDGE_MODEL")
        or "gemini-3.6-flash",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--force-enable",
        action="store_true",
        help="временно MO_ICD_LLM_REVIEW=1 для этого запуска",
    )
    args = ap.parse_args()

    if args.force_enable:
        os.environ["MO_ICD_LLM_REVIEW"] = "1"
    if not icd_llm_review_enabled() and not args.dry_run:
        print("MO_ICD_LLM_REVIEW is off; use --force-enable or --dry-run", file=sys.stderr)
        return 2

    day = _resolve_date(args.date)
    d = date.fromisoformat(day)
    out = args.out or (
        args.medical_exams_root
        / "llm_icd_review"
        / f"{d:%Y}"
        / f"{d:%m}"
        / f"{d:%d}"
        / "reviews.jsonl"
    )
    cases = load_day_cases(args.medical_exams_root, day)
    # only grey zone
    pending: list[dict[str, Any]] = []
    for case in cases:
        pipe = evaluate_mo_icd_match(case)
        if pipe.get("needs_llm_review"):
            pending.append(case)
        if args.limit and len(pending) >= args.limit:
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = 0
    with out.open("w", encoding="utf-8") as fh:
        for case in pending:
            try:
                row = judge_case(
                    case,
                    model_name=args.model,
                    dry_run=args.dry_run,
                )
                if row.get("skipped"):
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                row = {
                    "visit_id": str(case.get("visit_id") or ""),
                    "error": type(exc).__name__,
                    "error_detail": str(exc)[:200],
                    "engine": ENGINE,
                }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "day": day,
                "candidates": len(pending),
                "ok": n_ok,
                "skipped": n_skip,
                "errors": n_err,
                "out": str(out),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
