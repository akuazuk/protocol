from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ingest_mo_lab_from_mis_tests",
    _ROOT / "scripts" / "ingest_mo_lab_from_mis_tests.py",
)
assert _SPEC and _SPEC.loader
_INGEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INGEST)
patient_key_for = _INGEST.patient_key_for
_month_starts = _INGEST._month_starts
_ensure_unique_rows = _INGEST._ensure_unique_rows
DDL = _INGEST.DDL


def test_patient_key_matches_mo_daily_formula() -> None:
    expected = hashlib.sha256(b"42").hexdigest()[:20]
    assert patient_key_for("42") == expected
    assert patient_key_for("") == ""
    assert len(patient_key_for("1001")) == 20


def test_month_chunks_cover_range() -> None:
    chunks = _month_starts(date(2025, 12, 19), date(2026, 2, 3))
    assert chunks[0] == (date(2025, 12, 19), date(2026, 1, 1))
    assert chunks[-1][1] == date(2026, 2, 3)
    cursor = chunks[0][0]
    for start, end in chunks:
        assert start == cursor
        assert start < end
        cursor = end
    assert cursor == date(2026, 2, 3)


def test_night_pipeline_appends_lab_warehouse() -> None:
    text = (_ROOT / "deploy" / "gcp-app" / "night_mis_pipeline.sh").read_text(encoding="utf-8")
    assert "ingest_mo_lab_from_mis_tests.py" in text
    assert "--skip-coverage" in text
    assert "lab ingest failed (non-fatal)" in text
    assert "gce_lab_${DAY}.json" in text
    assert 'write_lab_status "success"' in text
    assert "run_mo_lab_rollout_metrics.py" in text
    assert "lab rollout metrics failed (non-fatal)" in text
    assert 'sudo chown "$(whoami):$(whoami)" "$DATA/reports"' in text
    checker = (_ROOT / "deploy" / "gcp-app" / "check_gce_night_status.sh").read_text(
        encoding="utf-8"
    )
    assert "LAB_STATUS_FILE" in checker
    assert 'lab_status == "success"' in checker


def test_exact_duplicate_rows_are_removed_and_rejected() -> None:
    db = sqlite3.connect(":memory:")
    db.executescript(DDL)
    row = ("key", "2026-08-20", 1, 2, "ОАК", 3, "Hb", "132", "г/л")
    db.executemany("INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)", [row, row])
    _ensure_unique_rows(db)
    assert db.execute("SELECT COUNT(*) FROM fact_mo_lab").fetchone()[0] == 1
    db.execute("INSERT OR IGNORE INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)", row)
    assert db.execute("SELECT COUNT(*) FROM fact_mo_lab").fetchone()[0] == 1
