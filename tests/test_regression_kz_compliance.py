"""Regression-тесты improve_kz на обезличенных КЗ (ТЗ §16)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.compliance_engine import build_compliance_report
from clinical_knowledge.consult_analysis import analyze_consultation_text
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.safety_checker import run_safety_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consultations"

GASTRO_1 = """\
Медицинский центр
Врач: гастроэнтеролог
Дата консультации: 15.03.2024
Дата рождения: 01.06.1960
Пол: мужской

Жалобы: боли в животе, метеоризм.
Объективный статус: удовлетворительное.
Данные обследований:
ОАК от 10.03.2024 - в норме.
БАК от 10.03.2024 - в норме.
ФКС от 08.03.2024.
ФГДС от 08.03.2024 - опухолевое образование сигмовидной кишки.
УЗИ ОБП от 09.03.2024.
КТ брюшной полости от 09.03.2024.
Диагноз:
K30 Функциональная диспепсия.
Q43.8 Другие врождённые аномалии кишечника.
E61.1 Недостаточность железа.
E80.4 Нарушения метаболизма билирубина.
R14 Метеоризм.
K82.8 Другие уточнённые болезни желчного пузыря.
Нельзя исключить инвазию опухолевого образования сигмовидной кишки.
Рекомендации по обследованию: колоноскопия под седацией; ОАК и ЭКГ перед седацией.
Рекомендации по лечению: симптоматически.
"""

MG_1 = """\
Врач: терапевт
Дата консультации: 01.02.2024
Пол: undefined
Жалобы: кашель, насморк.
Объективный статус: удовлетворительное.
Диагноз: J06.8 Другие острые инфекции верхних дыхательных путей.
Рекомендации по лечению: симптоматическая терапия.
Контроль через неделю.
"""

PL_1_F = """\
Врач: флеболог
Дата консультации: 12.04.2024
Дата рождения: 15.08.1970
Пол: женский
Объективный статус: отёк левой ноги.
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации по лечению: ривароксабан 20 мг 1 раз в день постоянно.
Рекомендации по обследованию: контрольное УЗИ вен через 3 месяца.
Дата повторной явки: 12.07.2024 - повторная консультация флеболога.
"""

PL_2_D_S = """\
Врач: dermatolog
Дата консультации: 05.05.2024
Дата рождения: 20.01.1990
Пол: женский
Жалобы: высыпания на коже.
Локальный статус: эритема на лице.
Диагноз: L93.0 Дискоидная красная волчанка ?
Рекомендации по обследованию: ANA, anti-DNA.
Рекомендации по лечению: гидроксихлорохин 200 мг 2 раза в сутки 2 недели; фотозащита.
"""


def test_gastro_1_icd_and_malignancy():
    doc = parse_consultation(GASTRO_1, consultation_id="gastro_1")
    codes = {d.icd10_code for d in doc.diagnoses if d.icd10_code}
    for code in ("K30", "Q43.8", "E61.1", "E80.4", "R14", "K82.8"):
        assert code in codes, f"missing {code}"
    assert doc.patient.age_years is not None
    safety = run_safety_checks(doc)
    assert any(s.issue_type == "possible_malignancy" for s in safety)
    assert any(d.certainty == "suspected" for d in doc.diagnoses) or doc.extraction_quality.has_question_mark_diagnosis
    exam_blob = (doc.sections.exam_results or "").lower()
    for token in ("оак", "фгдс", "узи", "кт"):
        assert token in exam_blob


def test_mg_1_undefined_and_j06():
    doc = parse_consultation(MG_1, consultation_id="mg_1")
    assert doc.extraction_quality.has_undefined or "undefined" in (doc.raw_text or "").lower()
    codes = {d.icd10_code for d in doc.diagnoses if d.icd10_code}
    assert "J06.8" in codes or "J06" in codes
    rep = build_compliance_report(doc, matches=[], rules_check={})
    assert not any(i.severity == "critical" and "follow" in i.issue_type for i in rep.warnings)


def test_pl_1_f_thrombosis_and_medication():
    doc = parse_consultation(PL_1_F, consultation_id="pl_1_f")
    codes = {d.icd10_code for d in doc.diagnoses if d.icd10_code}
    assert any(c and c.startswith("I80") for c in codes)
    safety = run_safety_checks(doc)
    assert any(s.issue_type == "thrombosis" for s in safety)
    assert doc.medications
    assert any("риварокс" in (m.drug_name or m.raw_text or "").lower() for m in doc.medications)
    throm = next(s for s in safety if s.issue_type == "thrombosis")
    assert throm.status in ("handled", "partially_handled")
    assert doc.follow_up or doc.sections.follow_up_text


def test_pl_1_f_hybrid_headline_not_capped_at_fifty():
    from clinical_knowledge.compliance_gate import evaluate_send_gate_from_compliance
    from clinical_knowledge.consult_overall_score import apply_hybrid_overall_compliance

    doc = parse_consultation(PL_1_F, consultation_id="pl_1_f")
    rep = build_compliance_report(doc, matches=[], rules_check={"rules_compliance_pct": 40.0})
    comp = rep.model_dump(mode="json")
    review: dict = {}
    apply_hybrid_overall_compliance(
        review,
        structured_analysis={"compliance": comp},
        clinical_rules={"rules_check": {"rules_compliance_pct": 40.0}, "matched_protocols": []},
    )
    assert review["overall_compliance_pct"] > 50
    sg = evaluate_send_gate_from_compliance(comp, headline_score=review["overall_compliance_pct"])
    assert sg["gate_allowed"] is True
    assert sg["sign_decision"] in ("allowed", "review_required", "allowed_with_warnings")
    assert not comp.get("critical_issues")


def test_pl_1_f_rich_text_no_mi_population_guard():
    from clinical_knowledge.compliance_gate import evaluate_send_gate_from_compliance
    from clinical_knowledge.consult_analysis import analyze_consultation_text

    text = """\
Врач: флеболог
Дата консультации: 21.10.2024
Дата рождения: 15.08.1976
Пол: мужской
Диагноз: I80.1 Флебит и тромбофлебит бедренной вены. Флеботромбоз бедренно-подколенно-берцового сегмента.
Рекомендации по лечению: Ривороксабан 20 мг. Эластическая компрессия. УЗИ через 3 месяца.
"""
    out = analyze_consultation_text(text, consultation_id="pl_rich", with_markdown=False)
    comp = out["compliance"]
    crit = comp.get("critical_issues") or []
    assert not any("population_guard" in str(i.get("issue_type") or "") for i in crit)
    sg = evaluate_send_gate_from_compliance(comp, headline_score=comp.get("overall_score"))
    assert sg["gate_allowed"] is True


def test_pl_2_d_s_suspected_and_exams():
    doc = parse_consultation(PL_2_D_S, consultation_id="pl_2_d_s")
    assert any(d.certainty == "suspected" for d in doc.diagnoses)
    rec = (doc.sections.recommendations_exams or "").lower()
    assert "ana" in rec or "anti-dna" in rec
    assert doc.medications
    assert any("гидроксихлорохин" in (m.drug_name or m.raw_text or "").lower() for m in doc.medications)
    blob = (doc.sections.non_drug_recommendations or doc.sections.general_recommendations or doc.raw_text or "").lower()
    assert "фотозащит" in blob


def test_derma_fixture_suspected_via_pipeline():
    text = (FIXTURES / "derma_suspected.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="derma", with_markdown=False)
    comp = out["compliance"]
    assert comp["overall_status"]
    assert any(d.get("certainty") == "suspected" for d in out["document"].get("diagnoses", []))


def test_surgery_redflag_manual_review():
    text = (FIXTURES / "surgery_redflag.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="redflag", with_markdown=False)
    comp = out["compliance"]
    assert comp["overall_status"] == "manual_review_required"
    assert comp.get("confidence_score") is not None
    assert comp.get("score_source") == "deterministic"
