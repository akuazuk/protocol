"""Dermatology L93 - no false MRI/CT from substring matches."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.patient_exam_extraction import extract_exams_from_text, lab_exams
from clinical_knowledge.patient_medication_extraction import extract_medications_from_text
from clinical_knowledge.patient_review import run_patient_review

DERM_SNIPPET = Path(__file__).parent / "fixtures" / "dermatology_kz_l93.txt"
CLIENT_KZ = Path(__file__).resolve().parents[1] / "clients_consult" / "pl_2_d_s.pdf"


def _derm_text() -> str:
    if CLIENT_KZ.is_file():
        from clinical_knowledge.text_extract import extract_text_from_path

        txt = extract_text_from_path(CLIENT_KZ)
        if txt and len(txt) > 200:
            return txt
    return DERM_SNIPPET.read_text(encoding="utf-8")


def test_no_false_imaging_from_substrings() -> None:
    text = _derm_text()
    exams = extract_exams_from_text(text)
    imaging = [e for e in exams if e.get("category") == "imaging"]
    assert not imaging, f"unexpected imaging: {imaging}"
    labs = lab_exams(exams)
    assert labs
    assert not any(e.get("exam_type") == "CT" for e in exams)


def test_medications_not_fragment_names() -> None:
    text = _derm_text()
    meds = extract_medications_from_text(text)
    names = {m["name"].lower() for m in meds}
    assert "летке" not in names
    assert any("гидроксихлорохин" in n or "тридерм" in n for n in names)


def test_derm_report_no_mri_in_summary() -> None:
    out = run_patient_review(text=_derm_text(), consultation_id="t-derm-l93")
    pr = out["patient_report"]
    blob = (
        (pr.get("plain_summary_ru") or "")
        + " "
        + str(pr.get("clarification_points"))
        + " "
        + (pr.get("top_summary") or {}).get("main_takeaway_ru", "")
    ).lower()
    assert "мрт" not in blob
    assert "волчан" in blob or "l93" in blob.replace(" ", "") or "гидроксихлорохин" in blob.lower()
    understood = pr.get("understood_from_document") or []
    exam_rows = [x for x in understood if x.get("type") in ("exams", "labs")]
    assert exam_rows
    assert "кт" not in str(exam_rows).lower() or "анализ" in str(exam_rows).lower()
