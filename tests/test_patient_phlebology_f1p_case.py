"""F_1_p phlebology - not dermatologist visit despite anamnesis mention."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.patient_narrative import extract_complaint_phrase, extract_specialty_phrase
from clinical_knowledge.patient_review import run_patient_review

SNIPPET = Path(__file__).parent / "fixtures" / "phlebology_f1p_snippet.txt"
PDF = Path(__file__).resolve().parents[1] / "clients_consult" / "F_1_p.pdf"


def _f1p_text() -> str:
    if PDF.is_file():
        from clinical_knowledge.text_extract import extract_text_from_path

        txt = extract_text_from_path(PDF)
        if txt and len(txt) > 500:
            return txt
    return SNIPPET.read_text(encoding="utf-8")


def test_f1p_specialty_is_phlebologist_not_dermatologist() -> None:
    text = _f1p_text()
    who = extract_specialty_phrase(text, "phlebology")
    assert who == "флеболога"
    assert "дермат" not in who


def test_f1p_complaint_not_prefixed_with_patient_label() -> None:
    text = _f1p_text()
    complaint = extract_complaint_phrase(text)
    assert not complaint.lower().startswith("пациента")
    assert "отек" in complaint.lower() or "голен" in complaint.lower()


def test_f1p_patient_report_summary() -> None:
    out = run_patient_review(text=_f1p_text(), consultation_id="t-f1p-golden")
    pr = out["patient_report"]
    summary = (pr.get("plain_summary_ru") or "").lower()
    assert "флеболог" in summary
    assert "дерматовенерolog" not in summary
    assert pr.get("patient_context", {}).get("specialty") == "phlebology"
