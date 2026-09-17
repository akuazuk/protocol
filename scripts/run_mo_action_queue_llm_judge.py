#!/usr/bin/env python3
"""Batch LLM-судья A/B только для action-очереди МО «Вчера».

Примеры:
  python3 scripts/run_mo_action_queue_llm_judge.py --date 2026-08-04 --source render --dry-run
  python3 scripts/run_mo_action_queue_llm_judge.py --date yesterday --stages ab --source local --limit 0

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
    items: list[Any] = []
    if isinstance(block, dict):
        raw = block.get("items")
        if isinstance(raw, list) and raw:
            items = list(raw)
    if not items:
        # fallback: action_queue list (часто единственный источник в report.json)
        queue = report.get("action_queue") or []
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
    items: list[Any] = []
    if isinstance(block, dict):
        raw = block.get("items")
        if isinstance(raw, list) and raw:
            items = list(raw)
    if not items:
        # fallback: action_queue (report часто пишет очередь сюда, а action_cases=null)
        fallback = report.get("action_queue") or []
        if isinstance(fallback, list):
            items = list(fallback)
    return [
        it
        for it in items
        if isinstance(it, dict) and (it.get("case_id") or it.get("visit_id") or it.get("mis_id"))
    ]


def load_local_case_document(
    item: dict[str, Any], *, day: str, medical_root: Path
) -> dict[str, Any]:
    """Слоты из ночного jsonl, без HTTP на Render (там 503, и Gemini зря не зовём)."""
    case_id = str(item.get("case_id") or item.get("visit_id") or "").strip()
    mis_id = str(item.get("mis_id") or "").strip()
    y, m, _ = day.split("-")
    path = medical_root / "secure_cases" / y / m / f"kz_l1_{day}_cases.jsonl"
    if not path.is_file():
        return {"_error": f"no_local_cases:{path.name}", "case_id": case_id}
    match: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        vid = str(row.get("visit_id") or row.get("case_id") or "").strip()
        mid = str(row.get("mis_id") or row.get("id") or "").strip()
        if (case_id and vid == case_id) or (mis_id and mid == mis_id):
            match = row
            break
    if match is None:
        return {"_error": "local_case_not_found", "case_id": case_id}
    clinical = {
        "complaints": match.get("complaints"),
        "anamnesis": match.get("anamnesis_doctor") or match.get("anamnesis"),
        "anamnesis_doctor": match.get("anamnesis_doctor"),
        "objective_status": match.get("objective_status"),
        "exam_data": match.get("exam_data"),
        "clinical_diagnosis": match.get("clinical_diagnosis") or match.get("diagnosis_main_text"),
        "mkb_code_main": match.get("mkb_code_main"),
        "exam_recommendations": match.get("exam_recommendations"),
        "treatment_recommendations": match.get("treatment_recommendations"),
        "follow_up": match.get("dispensary_info") or match.get("return_date"),
        "age_years": match.get("age_years"),
    }
    return {
        "ok": True,
        "case_id": case_id,
        "mis_id": match.get("mis_id") or mis_id,
        "visit_id": match.get("visit_id") or case_id,
        "clinical": clinical,
        "document_kind": match.get("document_kind"),
        "age_years": match.get("age_years"),
    }


def document_ready_for_llm(document: dict[str, Any], pack: dict[str, Any]) -> str:
    """Пустая строка если можно звать Gemini, иначе причина skip."""
    err = document.get("_error") or pack.get("document_error")
    if err:
        return str(err)
    slots = pack.get("slots") if isinstance(pack.get("slots"), dict) else {}
    if not any(str(value or "").strip() for value in slots.values()):
        return "empty_clinical_slots"
    return ""


def fetch_case_document(case_id: str, *, base_url: str) -> dict[str, Any]:
    """Клинические слоты из JSON case-detail (HTML /document не подходит)."""
    token = _methodist_token()
    url = f"{base_url.rstrip('/')}/api/methodist/mo/cases/{case_id}"
    try:
        detail = _http_json(url, token=token, timeout=90.0)
    except HTTPError as e:
        return {"_error": f"HTTP {e.code}", "case_id": case_id}
    except URLError as e:
        return {"_error": str(e.reason)[:200], "case_id": case_id}
    except json.JSONDecodeError as e:
        return {"_error": f"json: {e}", "case_id": case_id}
    if not isinstance(detail, dict):
        return {"_error": "case_detail_not_object", "case_id": case_id}
    document = detail.get("document") if isinstance(detail.get("document"), dict) else {}
    clinical = document.get("clinical") if isinstance(document.get("clinical"), dict) else {}
    record = detail.get("record") if isinstance(detail.get("record"), dict) else {}
    if not clinical:
        return {
            "_error": "no_clinical_in_case_detail",
            "case_id": case_id,
            "record": record,
        }
    return {
        "ok": True,
        "case_id": case_id,
        "mis_id": document.get("mis_id") or record.get("mis_id"),
        "visit_id": document.get("visit_id") or record.get("visit_id") or case_id,
        "clinical": clinical,
        "document_kind": document.get("document_kind") or record.get("document_kind"),
        "age_years": record.get("age_years") or clinical.get("age_years"),
    }


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


def _generate_gemini(prompt: str, *, model_name: str) -> tuple[str, int, dict[str, Any]]:
    from clinical_knowledge.gemini_client import build_model
    from clinical_knowledge.gemini_lite import _extract_text, generate_lite_json_response
    from clinical_knowledge.gemini_model_config import resolve_gemini_model
    from clinical_knowledge.mo_llm_usage import response_usage

    resolved, _warn = resolve_gemini_model(model_name)
    model = build_model(resolved)
    t0 = time.perf_counter()
    resp = generate_lite_json_response(model, prompt)
    text = _extract_text(resp)
    prompt_tokens, completion_tokens = response_usage(resp)
    ms = int((time.perf_counter() - t0) * 1000)
    return text, ms, {
        "model": resolved,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _maybe_record_usage(
    warehouse: Path | None,
    item: dict[str, Any],
    day: str,
    tier: str,
    usage: dict[str, Any],
    latency_ms: int,
) -> None:
    if warehouse is None or not warehouse.is_file():
        return
    from clinical_knowledge.mo_llm_usage import record_llm_usage

    case_id = str(item.get("case_id") or item.get("visit_id") or "").strip()
    try:
        record_llm_usage(
            warehouse,
            run_id=f"action-judge-{day}",
            tier=tier,
            model=str(usage.get("model") or ""),
            case_id=case_id,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int(latency_ms),
            status="ok",
            usage_date=day or None,
        )
    except Exception:  # noqa: BLE001
        return


def judge_one(
    item: dict[str, Any],
    *,
    stages: str,
    model_name: str,
    base_url: str,
    dry_run: bool,
    source: str = "render",
    day: str = "",
    medical_root: Path | None = None,
    warehouse: Path | None = None,
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

    if source == "local":
        document = load_local_case_document(
            item, day=day, medical_root=medical_root or Path("/var/data/medical_exams")
        )
    else:
        document = fetch_case_document(case_id, base_url=base_url)
    pack = document_to_case_pack(item, document)
    skip = document_ready_for_llm(document, pack)
    if skip:
        row["error"] = f"document: {skip}"
        return row

    stage_a_obj: dict[str, Any] | None = None
    try:
        if "a" in stages:
            prompt_a = build_prompt_a(pack)
            text_a, ms_a, usage_a = _generate_gemini(prompt_a, model_name=model_name)
            row["latency_ms_a"] = ms_a
            _maybe_record_usage(warehouse, item, day, "action_a", usage_a, ms_a)
            try:
                stage_a_obj = validate_stage_a(extract_json_object(text_a), case_id=case_id)
            except (ValueError, json.JSONDecodeError):
                text_a, ms_a2, usage_a2 = _generate_gemini(prompt_a, model_name=model_name)
                row["latency_ms_a"] = int(ms_a) + int(ms_a2)
                _maybe_record_usage(warehouse, item, day, "action_a_retry", usage_a2, ms_a2)
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
            prompt_b = build_prompt_b(pack, digest)
            text_b, ms_b, usage_b = _generate_gemini(prompt_b, model_name=model_name)
            row["latency_ms_b"] = ms_b
            _maybe_record_usage(warehouse, item, day, "action_b", usage_b, ms_b)
            try:
                row["stage_b"] = validate_stage_b(extract_json_object(text_b), case_id=case_id)
            except (ValueError, json.JSONDecodeError):
                text_b, ms_b2, usage_b2 = _generate_gemini(prompt_b, model_name=model_name)
                row["latency_ms_b"] = int(ms_b) + int(ms_b2)
                _maybe_record_usage(warehouse, item, day, "action_b_retry", usage_b2, ms_b2)
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
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="макс. число кейсов из action-очереди; 0 = все",
    )
    ap.add_argument("--warehouse", type=Path, default=None)
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

    if args.limit and args.limit > 0:
        items = items[: args.limit]
    print(f"date={day} source={args.source} action_items={len(items)} stages={stages} dry_run={args.dry_run}")
    if not items:
        print("очередь пуста - нечего прогонять")
        return 0

    for it in items:
        print(
            f"  - case_id={it.get('case_id') or it.get('visit_id')} "
            f"sev={it.get('severity')} reason={(it.get('reason') or it.get('finding_title') or '')[:80]}"
        )

    warehouse = args.warehouse
    if warehouse is None:
        candidate = args.medical_exams_root / "warehouse" / "mo_analytics.sqlite"
        warehouse = candidate if candidate.is_file() else None

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
                source=args.source,
                day=day,
                medical_root=args.medical_exams_root,
                warehouse=warehouse,
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
