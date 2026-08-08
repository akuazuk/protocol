"""Фаза 3: центральные пороги и primary-флаги."""
from __future__ import annotations

from clinical_knowledge.mo_icd_directory_eval import icd_directory_primary_enabled
from clinical_knowledge.mo_icd_name_match import icd_name_match_primary_enabled
from clinical_knowledge.mo_icd_thresholds import name_ok, snapshot


def test_threshold_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_NAME_OK", "0.55")
    assert abs(name_ok() - 0.55) < 1e-9
    snap = snapshot()
    assert abs(snap["name_ok"] - 0.55) < 1e-9


def test_pipeline_in_primary_enables_both_axes(monkeypatch) -> None:
    monkeypatch.delenv("MO_ICD_NAME_IN_PRIMARY", raising=False)
    monkeypatch.delenv("MO_ICD_DIR_IN_PRIMARY", raising=False)
    monkeypatch.setenv("MO_ICD_PIPELINE_IN_PRIMARY", "1")
    assert icd_name_match_primary_enabled() is True
    assert icd_directory_primary_enabled() is True


def test_per_axis_primary_still_works(monkeypatch) -> None:
    monkeypatch.setenv("MO_ICD_PIPELINE_IN_PRIMARY", "0")
    monkeypatch.setenv("MO_ICD_NAME_IN_PRIMARY", "1")
    monkeypatch.setenv("MO_ICD_DIR_IN_PRIMARY", "0")
    assert icd_name_match_primary_enabled() is True
    assert icd_directory_primary_enabled() is False


def test_etalon_pack_size() -> None:
    from pathlib import Path

    path = Path("eval/mo_icd_pipeline/etalon_labels_v1.jsonl")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 20
