#!/usr/bin/env python3
"""Глубокий прогон истории (слой B) и опционально сильная модель (слой C).

Слой C только на GCE: MO_LLM_EXECUTION_HOST=gce RUN_HOST=gcp.
Официальный overall_pct не переписываем. Пишем jsonl на диск.

  docker exec -e MO_LLM_EXECUTION_HOST=gce -e RUN_HOST=gcp -e MO_DATA_ROOT=/var/data/medical_exams \\
    protocol-web python scripts/run_mo_history_deep.py --date yesterday --limit 20
  docker exec ... protocol-web python scripts/run_mo_history_deep.py --date yesterday --llm --limit 15
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_case_document import clinical_fields_from_row, load_case_source_row  # noqa: E402
from clinical_knowledge.mo_history_continuity import (  # noqa: E402
    evaluate_history_continuity,
    rank_for_deep_run,
)
from clinical_knowledge.mo_history_deep import (  # noqa: E402
    pick_episode_prior,
    shadow_history_credit_finding,
)
from clinical_knowledge.mo_patient_history_bundle import attach_bundle_to_case  # noqa: E402

MINSK = ZoneInfo("Europe/Minsk")
DEFAULT_MODEL = os.environ.get("MO_HISTORY_DEEP_MODEL") or "gemini-3.6-flash"


def _resolve_date(raw: str) -> str:
    text = (raw or "").strip().lower()
    today = datetime.now(MINSK).date()
    if text in {"yesterday", "вчера"}:
        return (today - timedelta(days=1)).isoformat()
    if text in {"today", "сегодня"}:
        return today.isoformat()
    date.fromisoformat(text)
    return text


def _clip(text: str, n: int = 400) -> str:
    value = (text or "").strip()
    return value if len(value) <= n else value[: n - 1] + "…"


def _select_candidates(db: sqlite3.Connection, day: str, limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT mis_id, visit_id, visit_date, patient_key, doctor_key, doctor_id,
               specialty, diagnosis_code, diagnosis_text, overall_pct,
               zone1_band, zone2a_band, zone2b_band, attention_primary,
               history_prior_n, history_tier
        FROM fact_mo_case
        WHERE visit_date=?
          AND document_kind IN ('clinical_visit','consultation')
        """,
        (day,),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        cont = evaluate_history_continuity(
            current_code=str(rec.get("diagnosis_code") or ""),
            current_text=str(rec.get("diagnosis_text") or ""),
            zones=rec,
            attention_primary=str(rec.get("attention_primary") or ""),
            overall_pct=rec.get("overall_pct"),
            history_prior_n=int(rec.get("history_prior_n") or 0),
            history_tier=str(rec.get("history_tier") or ""),
        )
        rec.update(
            {
                "case_id": str(rec.get("visit_id") or rec.get("mis_id") or ""),
                "deep_run_score": cont.get("deep_run_score") or 0,
                "deep_run_track": cont.get("deep_run_track"),
                "continuity": cont,
            }
        )
        if int(rec["deep_run_score"] or 0) <= 0:
            continue
        items.append(rec)
    items.sort(key=rank_for_deep_run)
    return items[: max(1, int(limit))]


def _layer_b(row: dict[str, Any], db: sqlite3.Connection) -> dict[str, Any]:
    case = {
        "patient_key": row.get("patient_key") or "",
        "visit_date": row.get("visit_date"),
        "doctor_id": row.get("doctor_id") or "",
        "doctor_key": row.get("doctor_key") or "",
        "specialty": row.get("specialty") or "",
        "diagnosis_code": row.get("diagnosis_code") or "",
        "mis_id": row.get("mis_id"),
        "visit_id": row.get("visit_id"),
    }
    bundle = attach_bundle_to_case(case, warehouse=db, force=True)
    deep = pick_episode_prior(
        history_bundle=bundle,
        current_code=str(row.get("diagnosis_code") or ""),
        current_text=str(row.get("diagnosis_text") or ""),
    )
    current = None
    try:
        src = load_case_source_row(
            str(row.get("visit_id") or row.get("mis_id") or ""),
            visit_date=str(row.get("visit_date") or "")[:10],
            mis_id=str(row.get("mis_id") or "") or None,
        )
        current = clinical_fields_from_row(src) if src else None
    except Exception:  # noqa: BLE001
        current = None
    finding = shadow_history_credit_finding(deep)
    return {
        "case_id": row.get("case_id"),
        "mis_id": row.get("mis_id"),
        "visit_date": row.get("visit_date"),
        "deep_run_track": row.get("deep_run_track"),
        "deep_run_score": row.get("deep_run_score"),
        "continuity": row.get("continuity"),
        "layer_b": {
            "prior_n_loaded": deep.get("prior_n_loaded"),
            "prior_visit_date": deep.get("prior_visit_date"),
            "already_slots": deep.get("already_slots"),
            "prior_slots": deep.get("prior_slots"),
        },
        "shadow_finding": finding,
        "current_slot_keys": list((current or {}).keys()),
        "current_clinical": current,
        "prior_clinical": deep.get("prior_clinical"),
    }


def _layer_c(payload: dict[str, Any], *, model: str) -> dict[str, Any]:
    from scripts.run_mo_action_queue_llm_judge import _generate_gemini
    from scripts.run_mo_calibration_blind_judge import assert_gce_live_contour
    from clinical_knowledge.mo_llm_action_judge import extract_json_object

    assert_gce_live_contour()
    current = payload.get("current_clinical") or {}
    prior = payload.get("prior_clinical") or {}
    prompt = (
        "Ты методист. Сравни текущий визит и предыдущий визит того же эпизода. "
        "Официальный балл не ставь. Верни JSON: "
        '{"verdict":"review|poor|acceptable|good","history_explains_gap":true|false,'
        '"need_today_exam":true|false,"need_today_plan":true|false,"summary_ru":"..."}. '
        "Не выдумывай факты. Не пиши ФИО и id.\n"
        f"Текущий: {_clip(json.dumps(current, ensure_ascii=False), 1800)}\n"
        f"Prior: {_clip(json.dumps(prior, ensure_ascii=False), 1800)}\n"
    )
    raw, _ = _generate_gemini(prompt, model_name=model)
    parsed = extract_json_object(raw)
    return {
        "model": model,
        "verdict": str(parsed.get("verdict") or ""),
        "history_explains_gap": bool(parsed.get("history_explains_gap")),
        "need_today_exam": bool(parsed.get("need_today_exam", True)),
        "need_today_plan": bool(parsed.get("need_today_plan", True)),
        "summary_ru": str(parsed.get("summary_ru") or "")[:400],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="yesterday")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--warehouse",
        default=os.environ.get("MO_WAREHOUSE")
        or "/var/data/medical_exams/warehouse/mo_analytics.sqlite",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    day = _resolve_date(args.date)
    db = sqlite3.connect(args.warehouse)
    db.row_factory = sqlite3.Row
    selected = _select_candidates(db, day, args.limit)
    print("selected", len(selected), "day", day, flush=True)
    out_rows: list[dict[str, Any]] = []
    for row in selected:
        item = _layer_b(dict(row), db)
        if args.llm:
            try:
                item["layer_c"] = _layer_c(item, model=args.model)
            except Exception as exc:  # noqa: BLE001
                item["layer_c"] = {"error": type(exc).__name__}
        item.pop("current_clinical", None)
        item.pop("prior_clinical", None)
        out_rows.append(item)
    db.close()
    out = Path(args.out) if args.out else Path(
        f"/var/data/medical_exams/history_deep/{day}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        out.chmod(0o600)
    except OSError:
        pass
    print("wrote", str(out), "n", len(out_rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
