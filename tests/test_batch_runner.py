"""Тесты batch_runner (ТЗ §6, §16)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.batch_runner import SUPPORTED_SUFFIXES, analyze_file, run_batch

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consultations"


def test_analyze_file_returns_compliance(tmp_path: Path):
    path = FIXTURES / "gastro_adult.txt"
    res = analyze_file(path, out_dir=tmp_path)
    assert "compliance" in res
    assert res["compliance"]["overall_status"]
    assert (tmp_path / "gastro_adult.json").exists()
    assert (tmp_path / "gastro_adult.md").exists()


def test_run_batch_writes_summary(tmp_path: Path):
    expected = len(
        [
            p
            for p in FIXTURES.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )
    summary = run_batch(FIXTURES, out_dir=tmp_path)
    assert summary["analyzed"] == expected
    assert (tmp_path / "batch_summary.csv").exists()
    assert (tmp_path / "batch_summary.md").exists()
    rows = summary["results"]
    assert any(r["file"] == "gastro_adult.txt" for r in rows)
    assert all("overall_status" in r for r in rows)
