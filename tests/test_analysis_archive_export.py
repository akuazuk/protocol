"""Тест автоэкспорта архива анализов."""
from __future__ import annotations

import json

from clinical_knowledge.analysis_archive import (
    build_snapshot,
    export_latest_path,
    maybe_auto_export,
    save_snapshot_with_export,
)


def _minimal_snap(i: int = 0) -> dict:
    return build_snapshot(
        full_text=f"текст консультации {i}",
        source_file=f"case_{i}.pdf",
        build_version="test",
        structured_analysis={
            "document": {"patient": {}, "doctor_specialty": "Терапевт"},
            "compliance": {"overall_status": "mostly_compliant", "overall_score": 80},
            "rubric_specifics": {"rubrics": []},
        },
    )


def test_auto_export_triggers_every_n(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSULT_ARCHIVE_DIR", str(tmp_path))
    monkeypatch.setenv("CONSULT_ARCHIVE_ANALYSES", "1")
    monkeypatch.setenv("CONSULT_ARCHIVE_EXPORT_EVERY", "3")
    monkeypatch.setenv("CONSULT_ARCHIVE_EXPORT_SIZE", "10")

    for i in range(3):
        save_snapshot_with_export(_minimal_snap(i), build_version="test")

    meta = maybe_auto_export(build_version="test")
    assert meta.get("just_exported") is True
    assert export_latest_path().is_file()
    lines = export_latest_path().read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["source_basename"] == "case_0.pdf"
