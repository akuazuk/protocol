"""Prior clinical lookup for MZ rubric dynamics."""
from __future__ import annotations

import csv
from pathlib import Path

from clinical_knowledge.mo_case_document import load_prior_clinical, resolve_prior_clinical_for_case


def test_load_prior_clinical_finds_earlier_day(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "medical_exams"
    day_dir = root / "secure_cases" / "2026" / "08"
    day_dir.mkdir(parents=True)
    earlier = day_dir / "mo_2026-08-01.csv"
    with earlier.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "patient_id", "date", "exam_recommendations", "treatment_recommendations", "document_kind"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "100",
                "patient_id": "p-1",
                "date": "2026-08-01",
                "exam_recommendations": "УЗИ",
                "treatment_recommendations": "Старое лечение",
                "document_kind": "consultation",
            }
        )
    monkeypatch.setattr(
        "clinical_knowledge.mo_case_document._medical_exam_roots",
        lambda: [root],
    )
    prior = load_prior_clinical(
        patient_id="p-1",
        visit_date="2026-08-03",
        exclude_ids={"200"},
    )
    assert prior is not None
    assert prior["visit_date"] == "2026-08-01"
    assert prior["clinical"]["exam_recommendations"] == "УЗИ"
    assert "patient_id" not in prior


def test_resolve_prior_skips_without_patient(monkeypatch) -> None:
    monkeypatch.setattr(
        "clinical_knowledge.mo_case_document.load_case_source_row",
        lambda *args, **kwargs: {"id": "1", "complaints": "боль"},
    )
    monkeypatch.setattr(
        "clinical_knowledge.mo_case_document._warehouse_case_meta",
        lambda case_id: {"visit_date": "2026-08-03", "mis_id": "1"},
    )
    assert resolve_prior_clinical_for_case("1", visit_date="2026-08-03") is None
