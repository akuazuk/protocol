"""Tests for export_training_feedback.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_training_feedback import export_all  # noqa: E402


def test_export_seed_only_creates_datasets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("export_training_feedback.DATASETS_DIR", tmp_path)
    manifest = export_all(seed_only=True)
    assert manifest["counts"]["retrieval_pairs"] > 0
    assert (tmp_path / "retrieval_pairs.jsonl").is_file()
    assert (tmp_path / "kz_regression.jsonl").is_file()
    assert (tmp_path / "export_manifest.json").is_file()
    saved = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert saved["seed_only"] is True
