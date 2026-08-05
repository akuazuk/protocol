#!/usr/bin/env python3
"""Batch LLM-судья A/B только для action-очереди МО «Вчера».

Примеры:
  python3 scripts/run_mo_action_queue_llm_judge.py --date 2026-08-04 --source render --dry-run
  python3 scripts/run_mo_action_queue_llm_judge.py --date yesterday --stages ab --source render --limit 20

ПДн: out только под data/medical_exams/ или /var/data/medical_exams/ (не коммитить).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_llm_action_judge import (  # noqa: E402
    EXAMPLE_STAGE_A,
    EXAMPLE_STAGE_B,
    build_prompt_a,
    build_prompt_b,
    extract_json_object,
    stage_a_digest,
    validate_stage_a,
    validate_stage_b,
)

MINSK = ZoneInfo("Europe/Minsk")
DEFAULT_PROD = "https://protocol-bimy.onrender.com"


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


def _methodist_token() -> str:
    return (os.environ.get("METHODIST_TOKEN") or "").strip()


def _http_json(url: str, *, token: str | None = None, timeout: float = 60.0) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Methodist-Token"] = token
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_action_items_render(day: str, *, base_url: str) -> list[dict[str, Any]]:
    token = _methodist_token()
    if not token:
        raise SystemExit("METHODIST_TOKEN не задан (.env) - нужен для --source render")
    report = _http_json(
        f"{base_url.rstrip('/')}/api/methodist/mo/daily-report?date={day}",
        token=token,
    )
    block = report.get("action_cases") or {}
    items = block.get("items") if isinstance(block, dict) else None
    if not isinstance(items, list):
        # fallback: action_queue list
        queue = report.get("action_queue") or []
        items = []
        for q in queue:
            if not isinstance(q, dict):
                continue
            items.append(
                {
                    "case_id": str(q.get("visit_id") or q.get("case_id") or ""),
                    "mis_id": str(q.get("mis_id") or ""),
                    "severity": str(q.get("priority") or ""),
                    "reason": str(q.get("reason") or ""),
                    "overall_pct": q.get("score"),
                }
            )
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("case_id") or it.get("visit_id") or "").strip()
        if not cid:
            continue
        out.append(it)
    return out


def load_action_items_local(day: str, *, medical_root: Path) -> list[dict[str, Any]]:
    y, m, d = day.split("-")
    path = medical_root / "reports" / y / m / d / "report.json"
    if not path.is_file():
        raise SystemExit(f"нет локального отчёта: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    block = report.get("action_cases") or {}
    items = block.get("items") if isinstance(block, dict) else report.get("action_queue") or []
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict) and (it.get("case_id") or it.get("visit_id"))]


def fetch_case_document(case_id: str, *, base_url: str) -> dict[str, Any]:
    token = _methodist_token()
    url = f"{base_url.rstrip('/')}/api/methodist/mo/cases/{case_id}/document"
    try:
        return _http_json(url, token=token, timeout=90.0)
    except HTTPError as e:
        return {"_error": f"HTTP {e.code}", "case_id": case_id}
    except URLError as e:
        return {"_error": str(e.reason)[:200], "case_id": case_id}


def document_to_case_pack(item: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """Достаёт слоты из document API / secure payload без сырого result."""
    clinical = document.get("clinical") if isinstance(document.get("clinical"), dict) else {}
    if not clinical and isinstance(document.get("slots"), dict):
        clinical = document["slots"]
    # частые ключи в MO document payload
    detail = document.get("detail") if isinstance(document.get("detail"), dict) else {}
    src = {**detail, **clinical, **document}

    def g(*keys: str) -> str:
        for k in keys:
            v = src.get(k)
            if v is None and isinstance(src.get("fields"), dict):
                v = src["fields"].get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float)):
                return str(v)
        return ""

    case_id = str(item.get("case_id") or item.get("visit_id") or "").strip()
    meta = {
        "case_id": case_id,
        "visit_id": case_id,
        "mis_id": str(item.get("mis_id") or "").strip(),
        "queue_severity": str(item.get("severity") or item.get("queue_severity") or ""),
        "queue_reason": str(item.get("reason") or item.get("finding_title") or "")[:300],
        "overall_pct_system": item.get("overall_pct"),
        "age_years": src.get("age_years") or src.get("patient_age"),
    }
    slots = {
        "complaints": g("complaints", "complaint"),
        "anamnesis": g("anamnesis", "anamnesis_doctor", "anamnesis_auto"),
        "objective_status": g("objective_status", "status_localis", "objective"),
        "exam_data": g("exam_data", "investigations", "exam_results"),
        "clinical_diagnosis": g("clinical_diagnosis", "diagnosis", "diagnosis_main_text"),
        "mkb_code_main": g("mkb_code_main", "diagnosis_code", "icd_main"),
        "exam_recommendations": g("exam_recommendations", "recommendations_exam"),
        "treatment_recommendations": g("treatment_recommendations", "recommendations_treatment"),
        "follow_up": g("follow_up", "dispensary_info", "return_date"),
    }
    return {"meta": meta, "slots": slots, "document_error": document.get("_error")}


def _generate_gemini(prompt: str, *, model_name: str) -> tuple[str, int]:
    from clinical_knowledge.gemini_lite import get_lite_gemini_model
    from clinical_knowledge.gemini_model_config import resolve_gemini_model

    resolved, _warn = resolve_gemini_model(model_name)
    # temporarily prefer requested model via env for lite helper
    prev = os.environ.get("GEMINI_MODEL")
    os.environ["GEMINI_MODEL"] = resolved
    try:
        model = get_lite_gemini_model()
        if model is None:
            raise RuntimeError("Gemini model unavailable (нет ключа?)")
        t0 = time.perf_counter()
        # gemini_lite generate path
        from clinical_knowledge import gemini_lite

        if hasattr(gemini_lite, "generate_text"):
            text = gemini_lite.generate_text(prompt, max_out=4096)
        else:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or ""
            if not text:
                parts = []
                for c in getattr(resp, "candidates", None) or []:
                    content = getattr(c, "content", None)
                    for p in getattr(content, "parts", None) or []:
                        if getattr(p, "text", None):
                            parts.append(p.text)
                text = "".join(parts)
        ms = int((time.perf_counter() - t0) * 1000)
        return text, ms
    finally:
        if prev is None:
            os.environ.pop("GEMINI_MODEL", None)
        else:
            os.environ["GEMINI_MODEL"] = prev


def judge_one(
    item: dict[str, Any],
    *,
    stages: str,
    model_name: str,
    base_url: str,
    dry_run: bool,
) -> dict[str, Any]:
    case_id = str(item.get("case_id") or item.get("visit_id") or "").strip()
    row: dict[str, Any] = {
        "case_id": case_id,
        "visit_id": case_id,
        "mis_id": str(item.get("mis_id") or ""),
        "queue_reason": str(item.get("reason") or item.get("finding_title") or ""),
        "queue_severity": str(item.get("severity") or ""),
        "model_a": model_name if "a" in stages else None,
        "model_b": model_name if "b" in stages else None,
        "latency_ms_a": None,
        "latency_ms_b": None,
        "stage_a": None,
        "stage_b": None,
        "error": None,
    }
    if dry_run:
        pack = document_to_case_pack(item, {})
        pack["meta"]["case_id"] = case_id
        row["dry_run"] = True
        row["prompt_a_chars"] = len(build_prompt_a(pack)) if "a" in stages else 0
        digest = stage_a_digest(validate_stage_a(EXAMPLE_STAGE_A, case_id=case_id))
        row["prompt_b_chars"] = len(build_prompt_b(pack, digest)) if "b" in stages else 0
        return row

    document = fetch_case_document(case_id, base_url=base_url)
    pack = document_to_case_pack(item, document)
    if pack.get("document_error"):
        row["error"] = f"document: {pack['document_error']}"
        return row

    stage_a_obj: dict[str, Any] | None = None
    try:
        if "a" in stages:
            text_a, ms_a = _generate_gemini(build_prompt_a(pack), model_name=model_name)
            row["latency_ms_a"] = ms_a
            stage_a_obj = validate_stage_a(extract_json_object(text_a), case_id=case_id)
            row["stage_a"] = stage_a_obj
        if "b" in stages:
            if stage_a_obj is None:
                # B-only: минимальный digest из очереди
                digest = {
                    "diagnosis_score_pct": item.get("overall_pct"),
                    "diagnosis_verdict": "review",
                    "key_gaps": [],
                    "conclusion_ru": "",
                    "patient": {},
                }
            else:
                digest = stage_a_digest(stage_a_obj)
            text_b, ms_b = _generate_gemini(build_prompt_b(pack, digest), model_name=model_name)
            row["latency_ms_b"] = ms_b
            row["stage_b"] = validate_stage_b(extract_json_object(text_b), case_id=case_id)
    except Exception as e:  # noqa: BLE001 - batch must continue
        row["error"] = str(e)[:400]
    return row


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="LLM judge A/B for MO action queue only")
    ap.add_argument("--date", default="yesterday", help="YYYY-MM-DD | yesterday")
    ap.add_argument("--source", choices=("render", "local"), default="render")
    ap.add_argument("--base-url", default=os.environ.get("PROTOCOL_PROD_URL") or DEFAULT_PROD)
    ap.add_argument("--medical-exams-root", type=Path, default=ROOT / "data" / "medical_exams")
    ap.add_argument("--stages", default="ab", help="a | b | ab")
    ap.add_argument("--model", default=os.environ.get("MO_LLM_ACTION_JUDGE_MODEL") or "gemini-3.6-flash")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true", help="validate example fixtures and exit")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.self_check:
        a = validate_stage_a(EXAMPLE_STAGE_A)
        b = validate_stage_b(EXAMPLE_STAGE_B)
        print(json.dumps({"ok": True, "a_score": a["diagnosis_assessment"]["score_pct"], "b_score": b["plan_assessment"]["score_pct"]}, ensure_ascii=False))
        return 0

    day = _resolve_date(args.date)
    stages = "".join(ch for ch in args.stages.lower() if ch in "ab")
    if stages not in {"a", "b", "ab"}:
        raise SystemExit("--stages должен быть a, b или ab")

    if args.source == "render":
        items = load_action_items_render(day, base_url=args.base_url)
    else:
        items = load_action_items_local(day, medical_root=args.medical_exams_root)

    items = items[: max(0, args.limit)]
    print(f"date={day} source={args.source} action_items={len(items)} stages={stages} dry_run={args.dry_run}")
    if not items:
        print("очередь пуста - нечего прогонять")
        return 0

    for it in items:
        print(
            f"  - case_id={it.get('case_id') or it.get('visit_id')} "
            f"sev={it.get('severity')} reason={(it.get('reason') or it.get('finding_title') or '')[:80]}"
        )

    results: list[dict[str, Any]] = []
    conc = 1 if args.dry_run else max(1, min(args.concurrency, 6))
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [
            pool.submit(
                judge_one,
                it,
                stages=stages,
                model_name=args.model,
                base_url=args.base_url,
                dry_run=args.dry_run,
            )
            for it in items
        ]
        for fut in as_completed(futs):
            row = fut.result()
            row["date"] = day
            results.append(row)
            status = "dry" if args.dry_run else ("ERR" if row.get("error") else "ok")
            print(f"[{status}] {row.get('case_id')} a_ms={row.get('latency_ms_a')} b_ms={row.get('latency_ms_b')} err={row.get('error')}")

    results.sort(key=lambda r: str(r.get("case_id") or ""))
    out = args.out
    if out is None:
        y, m, d = day.split("-")
        out = args.medical_exams_root / "llm_action_judge" / y / m / d / "judges.jsonl"
    if not args.dry_run or args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(results)} rows -> {out}")
    else:
        print("dry-run: файл не писали (укажите --out чтобы сохранить манифест)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
