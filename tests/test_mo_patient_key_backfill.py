from __future__ import annotations

import sqlite3

from scripts.backfill_mo_patient_keys_from_mis import apply_keys, patient_key_for


def test_patient_key_is_truncated_sha256() -> None:
    key = patient_key_for("42")
    assert len(key) == 20
    assert patient_key_for("42") == key
    assert patient_key_for("") == ""
    assert patient_key_for("  ") == ""


def test_apply_keys_updates_empty_only() -> None:
    db = sqlite3.connect(":memory:")
    db.execute(
        """
        CREATE TABLE fact_mo_case (
            mis_id TEXT PRIMARY KEY,
            visit_id TEXT,
            patient_key TEXT
        )
        """
    )
    db.executemany(
        "INSERT INTO fact_mo_case VALUES (?,?,?)",
        [
            ("m1", "v1", ""),
            ("m2", "v2", "already"),
            ("m3", "v-missing", ""),
        ],
    )
    stats = apply_keys(
        db,
        by_visit={"v1": "key-visit", "v2": "ignored"},
        by_mis={"m3": "key-mis"},
    )
    rows = {
        row[0]: (row[1], row[2])
        for row in db.execute("SELECT mis_id, visit_id, patient_key FROM fact_mo_case")
    }
    assert stats["updated_by_visit"] == 1
    assert stats["updated_by_mis"] == 1
    assert rows["m1"][1] == "key-visit"
    assert rows["m2"][1] == "already"
    assert rows["m3"][1] == "key-mis"
