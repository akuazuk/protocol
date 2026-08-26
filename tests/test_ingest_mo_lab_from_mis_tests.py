from __future__ import annotations

import hashlib
import importlib.util
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
