"""End-to-end тест структурного анализа КЗ на обезличенных fixtures (ТЗ раздел 24)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_analysis import analyze_consultation_text

FIX = Path(__file__).parent / "fixtures" / "consultations"


def _analyze(name: str):
    text = (FIX / name).read_text(encoding="utf-8")
    return analyze_consultation_text(text, consultation_id=name, with_markdown=True)


def test_gastro_adult_pipeline():
    res = _analyze("gastro_adult.txt")
    doc = res["document"]
    assert doc["doctor_specialty"] and "гастро" in doc["doctor_specialty"].lower()
    assert doc["patient"]["age_years"] == 48
    assert any(d.get("icd10_code") == "K29.7" for d in doc["diagnoses"])
    assert len(doc["medications"]) >= 1
    assert res["compliance"]["overall_status"] in (
        "compliant", "mostly_compliant", "partially_compliant",
        "non_compliant", "insufficient_data", "manual_review_required",
    )
    assert "# Оценка консультативного заключения" in res["report_markdown"]


def test_derma_suspected_pipeline():
    res = _analyze("derma_suspected.txt")
    doc = res["document"]
    assert doc["patient"]["sex"] == "female"
    # подозрительный диагноз
    assert any(d.get("certainty") == "suspected" for d in doc["diagnoses"])
    # дообследование назначено -> safety по аутоиммунному флагу обработан
    assert res["compliance"]["overall_status"] != "manual_review_required"


def test_surgery_redflag_manual_review():
    res = _analyze("surgery_redflag.txt")
    # критический онко-флаг без маршрутизации -> manual_review_required
    assert res["compliance"]["overall_status"] == "manual_review_required"
    assert res["compliance"]["critical_issues"]


def test_batch_resilience_bad_input():
    # пустой/мусорный вход не должен ронять анализ (ТЗ 4.6)
    res = analyze_consultation_text("", consultation_id="empty")
    assert res["compliance"]["overall_status"] in ("insufficient_data", "manual_review_required")
    res2 = analyze_consultation_text("undefined\n???", consultation_id="junk")
    assert "document" in res2
