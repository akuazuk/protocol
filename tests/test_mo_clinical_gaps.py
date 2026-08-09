"""Clinical gaps for case-review parity (mo_1_test patterns)."""
from __future__ import annotations

from clinical_knowledge.mo_clinical_gaps import evaluate_mo_clinical_gaps


def _mo1_case() -> dict:
    return {
        "complaints": (
            "на кашель сухой, насморк. Борозды на ногтях. Головная боль, чаще по вечерам. "
            "Неправильная установка стоп при ходьбе. Набор веса неуточненный, по питанию"
        ),
        "anamnesis_doctor": (
            "Семейная АГ, гиперхолестеринемия. Хронические заболевания: "
            "Бронхиальная астма с 4х лет (ремиссия 1 год)."
        ),
        "objective_status": (
            "Кашля нет. Носовое дыхание не затруднено. Костно-мышечная система без отклонений. "
            "Кожа без изменений. Дыхание везикулярное, хрипы нет. ped at scab abs"
        ),
        "clinical_diagnosis": (
            "J45 Бронхиальная астма, аллергическая, контролируемая. "
            "Персистирующий аллергический ринит. М21.4 Плоско-вальгусная установка стоп? "
            "Е 55.0 Дефицит витамина Д? G44.2 Головная боль напряжения?"
        ),
        "exam_recommendations": "Аллергопанель, спирограмма, аллерголог",
        "treatment_recommendations": "витамин Д; явка с результатами",
    }


def test_mo1_triggers_core_gaps() -> None:
    findings = evaluate_mo_clinical_gaps(_mo1_case())
    codes = {f["code"] for f in findings}
    assert "B_complaint_exam_mismatch" in codes
    assert "B_dx_not_in_exam" in codes or "B_tentative_dx_weak_support" in codes
    assert "B_chronic_dx_therapy_absent" in codes
    assert "A_text_noise" in codes
    assert "B_treatment_before_confirmed_dx" in codes
    assert "B_complaint_not_addressed_in_plan" in codes


def test_clean_case_no_false_gaps() -> None:
    case = {
        "complaints": "боль в горле 2 дня",
        "anamnesis_doctor": "ОРВИ ранее",
        "objective_status": "Зев гиперемирован, налётов нет. Дыхание везикулярное.",
        "clinical_diagnosis": "J02.9 Острый фарингит",
        "exam_recommendations": "мазок из зева",
        "treatment_recommendations": "полоскание, наблюдение",
    }
    codes = {f["code"] for f in evaluate_mo_clinical_gaps(case)}
    assert "B_complaint_exam_mismatch" not in codes
    assert "B_chronic_dx_therapy_absent" not in codes
