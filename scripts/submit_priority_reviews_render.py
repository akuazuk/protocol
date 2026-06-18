#!/usr/bin/env python3
"""L1 + AI-review + analysis_review на Render (как «Одобрить» в кабинете методиста)."""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

DEFAULT_CASES = ("report_n_1", "report_n_2", "kard_1", "gastro_1", "F_1_p")
CLIENTS = ROOT / "clients_consult"


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _request(
    base: str,
    token: str,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    timeout: int = 300,
) -> dict:
    url = f"{base.rstrip('/')}{path}"
    data = None
    headers = {
        "X-Methodist-Token": token,
        "X-Methodist-Reviewer": os.environ.get("METHODIST_REVIEWER", "cursor-batch"),
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {path}: {err[:500]}") from e
    if not raw.strip():
        return {}
    out = json.loads(raw)
    return out if isinstance(out, dict) else {"data": out}


def _load_case_text(case_id: str) -> str:
    import io

    for ext in (".pdf", ".txt"):
        p = CLIENTS / f"{case_id}{ext}"
        if not p.is_file():
            continue
        if ext == ".txt":
            return p.read_text(encoding="utf-8", errors="replace").strip()
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Установите pypdf: pip install pypdf") from exc
        reader = PdfReader(io.BytesIO(p.read_bytes()))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    raise FileNotFoundError(f"Нет файла {case_id} в {CLIENTS}")


_BLOCK_ROWS = (
    ("documentation_score", ("structural_score", "documentation_quality_score"), "Оформление КЗ"),
    ("patient_data_score", (), "Данные пациента"),
    ("protocol_applicability_score", ("protocol_match_score",), "Применимость протокола"),
    ("diagnosis_score", (), "Диагноз"),
    ("required_exams_score", (), "Обследования"),
    ("treatment_score", (), "Лечение"),
    ("safety_score", (), "Безопасность"),
    ("follow_up_score", (), "Контроль"),
)


def _structured_blocks(result: dict) -> dict[str, dict]:
    bd = (((result.get("structured_analysis") or {}).get("compliance") or {}).get("score_breakdown") or {})
    blocks: dict[str, dict] = {}
    for key, fallbacks, label in _BLOCK_ROWS:
        score = None
        for k in (key, *fallbacks):
            v = bd.get(k)
            if isinstance(v, (int, float)):
                score = float(v)
                break
        blocks[key] = {"key": key, "label_ru": label, "score_pct": score}
    return blocks


def _findings_map(result: dict) -> dict[str, dict]:
    rc = ((result.get("clinical_rules") or {}).get("rules_check") or {})
    m: dict[str, dict] = {}
    for f in rc.get("findings") or []:
        rid = str(f.get("rule_id") or "")
        if rid:
            m[rid] = f
    return m


def _overrides_from_ai(ai: dict, result: dict) -> list[dict]:
    by_id = _findings_map(result)
    out: list[dict] = []
    for ro in ai.get("rule_overrides") or []:
        if not isinstance(ro, dict) or not ro.get("rule_id") or ro.get("human_pass") is None:
            continue
        f = by_id.get(str(ro["rule_id"])) or {}
        sys_pass = f.get("passed")
        sys_pass = bool(sys_pass) if sys_pass is not None else None
        human_pass = bool(ro["human_pass"])
        if sys_pass is human_pass:
            continue
        out.append(
            {
                "rule_id": str(ro["rule_id"]),
                "system_pass": sys_pass,
                "human_pass": human_pass,
                "note": str(ro.get("note") or "")[:280],
            }
        )
    return out


def _block_overrides_from_ai(ai: dict, result: dict) -> list[dict]:
    blocks = _structured_blocks(result)
    out: list[dict] = []
    for bo in ai.get("block_overrides") or []:
        if not isinstance(bo, dict) or bo.get("verdict") != "disagree" or not bo.get("block_key"):
            continue
        b = blocks.get(bo["block_key"]) or {}
        out.append(
            {
                "block_key": str(bo["block_key"]),
                "block_label_ru": b.get("label_ru") or bo["block_key"],
                "system_score": b.get("score_pct"),
                "human_agrees": False,
                "note": str(bo.get("note") or "")[:280],
            }
        )
    return out


def _review_note(ai: dict) -> str:
    parts: list[str] = []
    if ai.get("summary_ru"):
        parts.append(str(ai["summary_ru"]))
    if ai.get("system_notes_ru"):
        parts.append("Ошибки системы: " + str(ai["system_notes_ru"]))
    items = ai.get("engine_improvements_ru") or ai.get("improvements_ru") or []
    if items:
        parts.append("Правки движка: " + "; ".join(str(x) for x in items))
    return "\n\n".join(parts)[:2000]


def submit_one(base: str, token: str, case_id: str, *, dry_run: bool = False) -> dict:
    text = _load_case_text(case_id)
    if dry_run:
        return {"case_id": case_id, "dry_run": True, "text_len": len(text)}

    t0 = time.perf_counter()
    result = _request(
        base,
        token,
        "POST",
        "/api/consult-review/tier",
        {
            "tier": "L1",
            "text": text,
            "consultation_id": case_id,
            "methodist_mode": True,
            "category_slugs": "",
        },
        timeout=360,
    )
    analysis_id = str(result.get("analysis_id") or "")
    text_hash = str(result.get("text_hash") or "")
    if not analysis_id:
        raise RuntimeError(f"{case_id}: нет analysis_id в ответе tier")

    ai_resp = _request(
        base,
        token,
        "POST",
        "/api/methodist/ai-review",
        {"analysis_id": analysis_id},
        timeout=180,
    )
    ai = ai_resp.get("ai_review") or {}
    if not ai:
        raise RuntimeError(f"{case_id}: пустой ai_review")

    body: dict = {
        "event_type": "analysis_review",
        "analysis_id": analysis_id,
        "text_hash": text_hash,
        "rating": int(ai["system_accuracy_rating"]),
        "verdict": str(ai["system_accuracy_verdict"]),
        "kz_compliance_gold": str(ai.get("kz_compliance_gold") or ""),
        "tags": list(ai.get("tags") or []),
        "note": _review_note(ai),
        "reviewer": os.environ.get("METHODIST_REVIEWER", "cursor-batch"),
        "review_source": "ai_assisted",
        "methodist_approved": True,
        "overrides": _overrides_from_ai(ai, result),
        "block_overrides": _block_overrides_from_ai(ai, result),
        "ai_review": {
            "model_used": ai.get("model_used") or ai_resp.get("model_used") or "",
            "confidence": ai.get("confidence") or "",
            "summary_ru": ai.get("summary_ru") or "",
            "system_notes_ru": ai.get("system_notes_ru") or "",
            "engine_improvements_ru": ai.get("engine_improvements_ru") or [],
        },
    }
    rf = ai.get("retrieval_fix")
    if isinstance(rf, dict) and (rf.get("chosen_path") or "").strip():
        body["retrieval_fix"] = {
            "query": "",
            "rejected_path": str(rf.get("rejected_path") or ""),
            "chosen_path": str(rf["chosen_path"]),
        }

    fb = _request(base, token, "POST", "/api/ml/feedback", body, timeout=60)
    ms = int((time.perf_counter() - t0) * 1000)
    return {
        "case_id": case_id,
        "analysis_id": analysis_id,
        "text_hash": text_hash,
        "overall_pct": (result.get("review") or {}).get("overall_compliance_pct"),
        "rating": body["rating"],
        "verdict": body["verdict"],
        "tags": body["tags"],
        "event_id": fb.get("event_id"),
        "ms": ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("RENDER_URL", "https://protocol-bimy.onrender.com"))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument(
        "--from-report",
        type=Path,
        default=None,
        help="Все case_id из report.json batch",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "experiments" / "batch_clients_consult_2026-06-01" / "render_reviews.json")
    args = ap.parse_args()

    token = (os.environ.get("METHODIST_TOKEN") or os.environ.get("METHODIST_PIN") or "").strip()
    if not token and not args.dry_run:
        print("Задайте METHODIST_TOKEN в .env", file=sys.stderr)
        return 1

    if args.from_report:
        if not args.from_report.is_file():
            print(f"Report not found: {args.from_report}", file=sys.stderr)
            return 1
        data = json.loads(args.from_report.read_text(encoding="utf-8"))
        case_ids = [str(r["case_id"]) for r in (data.get("reports") or []) if r.get("case_id")]
    elif args.cases:
        case_ids = list(args.cases)
    else:
        case_ids = list(DEFAULT_CASES)

    results: list[dict] = []
    errors: list[dict] = []
    for case_id in case_ids:
        try:
            row = submit_one(args.base, token, case_id, dry_run=args.dry_run)
            results.append(row)
            print(f"OK {case_id} rating={row.get('rating')} verdict={row.get('verdict')} ({row.get('ms')} ms)")
        except Exception as exc:
            errors.append({"case_id": case_id, "error": str(exc)[:400]})
            print(f"ERR {case_id}: {exc}", file=sys.stderr)

    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"results": results, "errors": errors}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.out}")

    return 0 if results and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
