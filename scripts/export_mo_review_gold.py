#!/usr/bin/env python3
"""Export crm_review_pack rows with training_use=1 for offline eval.

Пример:
  python3 scripts/export_mo_review_gold.py \\
    --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite \\
    --out /var/data/medical_exams/gold_review/2026-08-05
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _hash_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warehouse", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-patient-plain", action="store_true")
    args = ap.parse_args()
    if not args.warehouse.is_file():
        raise SystemExit(f"warehouse_missing:{args.warehouse}")
    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.warehouse))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT pack_id, case_id, visit_id, mis_id, patient_id, visit_date,
                  doctor_fio, specialty, filial, clinical_json, system_json,
                  decision_json, training_use, actor, created_at, supersedes_pack_id
           FROM crm_review_pack
           WHERE training_use=1
           ORDER BY created_at"""
    ).fetchall()
    packs_path = args.out / "review_packs.jsonl"
    ratings_path = args.out / "protocol_ratings.jsonl"
    n_ratings = 0
    with packs_path.open("w", encoding="utf-8") as packs_out, ratings_path.open(
        "w", encoding="utf-8"
    ) as ratings_out:
        for row in rows:
            decision = json.loads(row["decision_json"] or "{}")
            system = json.loads(row["system_json"] or "{}")
            clinical = json.loads(row["clinical_json"] or "{}")
            patient = row["patient_id"] or ""
            payload = {
                "pack_id": row["pack_id"],
                "case_id": row["case_id"],
                "visit_id": row["visit_id"],
                "mis_id": row["mis_id"],
                "patient_id": patient if args.include_patient_plain else _hash_id(patient),
                "visit_date": row["visit_date"],
                "doctor_fio": row["doctor_fio"],
                "specialty": row["specialty"],
                "filial": row["filial"],
                "clinical": clinical,
                "system": {
                    "overall_pct": system.get("overall_pct"),
                    "findings": system.get("findings") or [],
                    "llm_action_judge": system.get("llm_action_judge") or {},
                    "protocol_suggest": system.get("protocol_suggest") or {},
                },
                "decision": decision,
                "actor": row["actor"],
                "created_at": row["created_at"],
                "supersedes_pack_id": row["supersedes_pack_id"],
            }
            packs_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for rating in decision.get("protocol_ratings") or []:
                if not isinstance(rating, dict):
                    continue
                ratings_out.write(
                    json.dumps(
                        {
                            "pack_id": row["pack_id"],
                            "case_id": row["case_id"],
                            "visit_date": row["visit_date"],
                            "protocol_id": rating.get("protocol_id"),
                            "title": rating.get("title"),
                            "relevance": rating.get("relevance"),
                            "note_ru": rating.get("note_ru"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_ratings += 1
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "packs": len(rows),
        "protocol_ratings": n_ratings,
        "warehouse": str(args.warehouse),
        "patient_id_mode": "plain" if args.include_patient_plain else "sha256_16",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
