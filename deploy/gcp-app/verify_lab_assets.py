"""Verify packaged lab canons and the numerical lab evaluator in a built image."""

from __future__ import annotations

import os
import sqlite3

from clinical_knowledge.lab_abnormal_findings import (
    CODE_ABNORMAL_IGNORED,
    load_reference_ranges,
)
from clinical_knowledge.lab_canons import lab_panels
from clinical_knowledge.mo_lab_shadow import evaluate_lab_for_case


def main() -> None:
    ranges = load_reference_ranges()
    panels = lab_panels()
    if not ranges:
        raise SystemExit("lab_reference_ranges.json is missing or empty")
    if not panels:
        raise SystemExit("lab_test_canons.json is missing or empty")

    os.environ["MO_LAB_BUNDLE"] = "1"
    os.environ["MO_LAB_ABNORMAL"] = "1"
    os.environ["MO_LAB_ABNORMAL_PRIMARY"] = "0"

    lab = sqlite3.connect(":memory:")
    try:
        lab.execute(
            """
            CREATE TABLE fact_mo_lab(
                patient_key TEXT,
                test_date TEXT,
                test_id INTEGER,
                type_id INTEGER,
                type_name TEXT,
                indicator_id INTEGER,
                indicator_name TEXT,
                value TEXT,
                unit TEXT
            )
            """
        )
        rows = [
            ("synthetic-patient", "2026-08-19", 1, 1, "Synthetic", 1, "Глюкоза", "12.5", "ммоль/л"),
            ("synthetic-patient", "2026-08-21", 2, 1, "Synthetic", 1, "Глюкоза", "13.0", "ммоль/л"),
            ("synthetic-other", "2026-08-19", 3, 1, "Synthetic", 1, "Глюкоза", "14.0", "ммоль/л"),
            ("synthetic-patient", "2026-08-19", 4, 1, "Synthetic", 2, "Глюкоза", "15.0", "моль/л"),
        ]
        lab.executemany("INSERT INTO fact_mo_lab VALUES(?,?,?,?,?,?,?,?,?)", rows)
        payload, findings = evaluate_lab_for_case(
            {
                "patient_key": "synthetic-patient",
                "visit_date": "2026-08-20",
                "age_years": 40,
            },
            lab_db=lab,
        )
    finally:
        lab.close()

    abnormal = [
        finding
        for finding in findings
        if finding.get("code") == CODE_ABNORMAL_IGNORED
    ]
    if len(abnormal) != 1 or abnormal[0].get("shadow") is not True:
        raise SystemExit("synthetic lab evaluation did not return one shadow finding")
    if payload.get("abnormal_check", {}).get("status") != "completed_limited":
        raise SystemExit("synthetic lab evaluation did not complete")

    print(
        "lab image verification ok: "
        f"ranges={len(ranges)} panels={len(panels)} shadow_findings={len(abnormal)}"
    )


if __name__ == "__main__":
    main()
