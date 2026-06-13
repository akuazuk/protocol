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
    # В tmp нет corpus_chunks - мокаем resolver для unit-теста export.
    fake_index = {"gastro/kp.pdf": "ГЭРБ клинический протокол"}

    def _fake_enrich(pairs):
        out = []
        for row in pairs:
            p = row.get("positive_path", "")
            if p in fake_index or row.get("query"):
                out.append({**row, "positive_path": p or "gastro/kp.pdf", "positive_text": fake_index["gastro/kp.pdf"]})
        stats = {"total": len(pairs), "resolved": len(out), "missing": len(pairs) - len(out), "with_negative": 0}
        return out, stats

    monkeypatch.setattr("export_training_feedback.enrich_retrieval_with_texts", _fake_enrich)
    manifest = export_all(seed_only=True)
    assert manifest["counts"]["retrieval_pairs"] > 0
    assert (tmp_path / "retrieval_pairs.jsonl").is_file()
    assert (tmp_path / "kz_regression.jsonl").is_file()
    assert (tmp_path / "export_manifest.json").is_file()
    assert (tmp_path / "retrieval_pairs_resolved.jsonl").is_file()
    saved = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert saved["seed_only"] is True
    assert saved["counts"]["retrieval_pairs_resolved"] > 0


def test_export_reads_ml_feedback_dir(tmp_path, monkeypatch) -> None:
    fb = tmp_path / "remote_feedback"
    fb.mkdir()
    review = {
        "event_type": "analysis_review",
        "analysis_id": "a-neo",
        "text_hash": "sha256:abc",
        "rating": 2,
        "verdict": "partially_wrong",
        "reviewer": "Pavel",
        "overrides": [
            {
                "rule_id": "9bdafb96_path_neoplasm_diagnosis_formula",
                "system_pass": False,
                "human_pass": True,
                "note": "ложное срабатывание",
            }
        ],
    }
    (fb / "analysis_review.jsonl").write_text(
        json.dumps(review, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    out = tmp_path / "datasets"
    monkeypatch.setenv("ML_FEEDBACK_DIR", str(fb))
    monkeypatch.setattr("export_training_feedback.DATASETS_DIR", out)
    monkeypatch.setattr(
        "export_training_feedback.enrich_retrieval_with_texts",
        lambda pairs: ([], {"total": 0, "resolved": 0, "missing": 0, "with_negative": 0}),
    )
    manifest = export_all(seed_only=False)
    assert manifest["counts"]["feedback_events"] == 1
    assert manifest["counts"]["entailment_pairs"] == 1
    assert manifest["sources"]["feedback_dir"] == str(fb)
    ent = json.loads((out / "entailment_pairs.jsonl").read_text(encoding="utf-8").strip())
    assert ent["term"] == "9bdafb96_path_neoplasm_diagnosis_formula"
    assert ent["label"] == "entailment"
