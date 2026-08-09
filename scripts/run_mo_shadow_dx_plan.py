#!/usr/bin/env python3
"""Compute conservative shadow Dx/Plan scores for clinical MO visits (GCE only)."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_llm_action_judge import extract_json_object  # noqa: E402
from clinical_knowledge.mo_shadow_dx_plan import (  # noqa: E402
    ENGINE,
    build_shadow_payload,
    shadow_jsonl_path,
)
from scripts.run_mo_calibration_blind_judge import (  # noqa: E402
    assert_gce_live_contour,
    build_dx_prompt,
    build_plan_prompt,
    pin_dx_semantics,
    pin_plan_route,
    protocol_context_for_case,
)
from clinical_knowledge.mo_dx_evidence_score import validate_dx_evidence_result  # noqa: E402
from clinical_knowledge.mo_plan_protocol_score import (  # noqa: E402
    validate_plan_concordance_result,
)

MINSK = ZoneInfo("Europe/Minsk")
DEFAULT_MODEL = "gemini-3.6-flash"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        if text.startswith("export "):
            text = text[7:].strip()
        key, value = text.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_date(raw: str) -> str:
    text = (raw or "").strip().lower()
    today = datetime.now(MINSK).date()
    if text in {"yesterday", "вчера"}:
        return (today - timedelta(days=1)).isoformat()
    if text in {"today", "сегодня"}:
        return today.isoformat()
    date.fromisoformat(text)
    return text


def _sanitize_blocked(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    verdict = str(out.get("verdict") or "").strip().lower()
    if verdict not in {"blocked", "na"}:
        return out
    if endpoint == "dx":
        out["dx_evidence_pct"] = None
    else:
        for key in (
            "exam_protocol_pct",
            "treatment_protocol_pct",
            "followup_protocol_pct",
            "plan_protocol_pct",
            "plan_general_llm_pct",
        ):
            out[key] = None
    return out


def _generate(prompt: str, *, model: str) -> tuple[str, int]:
    from scripts.run_mo_action_queue_llm_judge import _generate_gemini

    return _generate_gemini(prompt, model_name=model)


def _cases_path(day: str, *, data_root: Path) -> Path:
    y, m, _ = day.split("-")
    return data_root / "secure_cases" / y / m / f"kz_l1_{day}_cases.jsonl"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _is_clinical(row: Mapping[str, Any]) -> bool:
    kind = str(
        row.get("document_kind")
        or (row.get("source") or {}).get("document_kind")
        or row.get("visit_type")
        or ""
    ).strip().lower()
    if not kind:
        return True
    if kind in {"clinical_visit", "consultation", "kz", "ambulatory"}:
        return True
    # secure_cases usually already filtered; keep unknown
    if "nonclinical" in kind or "procedure" in kind or "stomat" in kind:
        return False
    return True


def _case_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("visit_id")
        or row.get("case_id")
        or row.get("mis_id")
        or (row.get("source") or {}).get("visit_id")
        or ""
    ).strip()


def _mis_id(row: Mapping[str, Any]) -> str:
    return str(row.get("mis_id") or (row.get("source") or {}).get("mis_id") or "").strip()


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            for key in (
                str(row.get("case_id") or "").strip(),
                str(row.get("visit_id") or "").strip(),
                str(row.get("mis_id") or "").strip(),
            ):
                if key:
                    out[key] = row
    return out


def _clinical_concordance(warehouse: Path | None, mis_id: str) -> float | None:
    if warehouse is None or not warehouse.is_file() or not mis_id:
        return None
    try:
        conn = sqlite3.connect(str(warehouse))
        try:
            row = conn.execute(
                """
                SELECT score FROM fact_mo_score_axis
                WHERE mis_id = ? AND axis = 'clinical_concordance'
                LIMIT 1
                """,
                (mis_id,),
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None
    return None


def judge_one(
    row: dict[str, Any],
    *,
    day: str,
    model: str,
    warehouse: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    case_id = _case_id(row)
    mis_id = _mis_id(row)
    concordance = _clinical_concordance(warehouse, mis_id)
    if dry_run:
        return build_shadow_payload(
            case_id=case_id,
            visit_date=day,
            model=model,
            dx_result={
                "verdict": "partial",
                "dx_evidence_pct": 60,
                "potential_harm": False,
                "summary_ru": "dry-run placeholder dx",
            },
            plan_result={
                "verdict": "partial",
                "plan_general_llm_pct": 60,
                "potential_harm": False,
                "summary_ru": "dry-run placeholder plan",
                "provenance": "llm_no_kp",
            },
            clinical_concordance_pct=concordance,
        ) | {"visit_id": case_id, "mis_id": mis_id, "dry_run": True}

    assert_gce_live_contour()
    from scripts.run_mo_calibration_blind_judge import blind_case_pack

    pack = blind_case_pack(row, sample_id=case_id or mis_id or "case")
    route, protocol_context = protocol_context_for_case(row, pack)
    dx_prompt, _ = build_dx_prompt(pack)
    plan_prompt, _ = build_plan_prompt(
        pack,
        route=route["route"],
        protocol_context=protocol_context,
    )
    dx_result = None
    plan_result = None
    error = None
    try:
        for attempt in range(2):
            try:
                raw, _ = _generate(dx_prompt, model=model)
                dx_result = validate_dx_evidence_result(
                    _sanitize_blocked("dx", pin_dx_semantics(extract_json_object(raw))),
                    case_id=case_id,
                )
                break
            except (ValueError, json.JSONDecodeError):
                if attempt == 1:
                    raise
                time.sleep(0.5)
        for attempt in range(2):
            try:
                raw, _ = _generate(plan_prompt, model=model)
                pinned = _sanitize_blocked(
                    "plan",
                    pin_plan_route(
                        extract_json_object(raw),
                        route=route["route"],
                        protocol_context=protocol_context
                        if isinstance(protocol_context, dict)
                        else None,
                    ),
                )
                plan_result = validate_plan_concordance_result(pinned, case_id=case_id)
                break
            except (ValueError, json.JSONDecodeError):
                if attempt == 1:
                    raise
                time.sleep(0.5)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    payload = build_shadow_payload(
        case_id=case_id,
        visit_date=day,
        model=model,
        dx_result=dx_result,
        plan_result=plan_result,
        clinical_concordance_pct=concordance,
        error=error,
    )
    payload["visit_id"] = case_id
    payload["mis_id"] = mis_id
    payload["route"] = route.get("route")
    return payload


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="yesterday")
    parser.add_argument(
        "--medical-exams-root",
        type=Path,
        default=Path(os.environ.get("MO_DATA_ROOT") or "/var/data/medical_exams"),
    )
    parser.add_argument("--warehouse", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("MO_SHADOW_DX_PLAN_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    day = _resolve_date(args.date)
    data_root = args.medical_exams_root.expanduser().resolve()
    warehouse = args.warehouse or (data_root / "warehouse" / "mo_analytics.sqlite")
    out = args.out or shadow_jsonl_path(day, root=data_root)
    cases_path = _cases_path(day, data_root=data_root)
    cases = [row for row in _load_cases(cases_path) if _is_clinical(row) and _case_id(row)]
    if args.limit > 0:
        cases = cases[: args.limit]

    existing = _load_existing(out) if args.resume else {}
    pending = []
    for row in cases:
        cid = _case_id(row)
        mid = _mis_id(row)
        if args.resume and (cid in existing or (mid and mid in existing)):
            prev = existing.get(cid) or existing.get(mid)
            if prev and not prev.get("error"):
                continue
        pending.append(row)

    print(
        json.dumps(
            {
                "date": day,
                "cases_path": str(cases_path),
                "cases_n": len(cases),
                "pending_n": len(pending),
                "resume": bool(args.resume),
                "dry_run": bool(args.dry_run),
                "model": args.model,
                "out": str(out),
                "engine": ENGINE,
            },
            ensure_ascii=False,
        )
    )
    if not args.dry_run:
        assert_gce_live_contour()

    results = list(existing.values()) if args.resume else []
    # de-dupe by case_id for rewrite
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        key = str(row.get("case_id") or row.get("visit_id") or "").strip()
        if key:
            by_id[key] = row

    conc = 1 if args.dry_run else max(1, min(args.concurrency, 4))
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [
            pool.submit(
                judge_one,
                row,
                day=day,
                model=args.model,
                warehouse=warehouse if warehouse.is_file() else None,
                dry_run=args.dry_run,
            )
            for row in pending
        ]
        for fut in as_completed(futures):
            payload = fut.result()
            key = str(payload.get("case_id") or payload.get("visit_id") or "").strip()
            if key:
                by_id[key] = payload
            band = payload.get("case_attention_band")
            print(
                f"case={key} band={band} err={bool(payload.get('error'))}",
                flush=True,
            )

    ordered = sorted(by_id.values(), key=lambda row: str(row.get("case_id") or ""))
    _atomic_write_jsonl(out, ordered)
    attention_n = sum(
        1 for row in ordered if str(row.get("case_attention_band") or "") in {"poor", "critical"}
    )
    print(
        json.dumps(
            {
                "written_n": len(ordered),
                "attention_n": attention_n,
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
