"""Тесты генерации отчётов (JSON + Markdown), ТЗ раздел 20."""
from __future__ import annotations

from clinical_knowledge.compliance_engine import build_compliance_report
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.consult_report import report_to_html, report_to_json, report_to_markdown

KZ = """\
Врач: гастроэнтеролог
Дата консультации: 14.07.2024
Дата рождения: 12.05.1976
Пол: мужской
Жалобы: боли в эпигастрии.
Анамнез: 2 года.
Объективный статус: удовлетворительное.
Диагноз: K29.7 Хронический гастрит.
Рекомендации по лечению: Омепразол 20 мг 2 раза в день 14 дней.
Дата повторной явки: 28.07.2024
"""


def _report():
    doc = parse_consultation(KZ, consultation_id="r1")
    matches = [{
        "protocol_id": "p1", "title": "КП гастрит", "source_path": "gastro/g.pdf",
        "population": "adult", "icd10_primary": ["K29.7"], "match_score": 80,
        "applicability": "applicable", "match_reasons": ["МКБ совпал"], "mismatch_reasons": [],
    }]
    return doc, build_compliance_report(doc, matches=matches, rules_check={})


def test_json_report_shape():
    doc, rep = _report()
    j = report_to_json(rep, doc)
    assert j["consultation_id"] == "r1"
    assert j["patient_summary"]["age_years"] == 48
    assert "score_breakdown" in j
    assert j["overall_status"] in (
        "compliant", "mostly_compliant", "partially_compliant",
        "non_compliant", "insufficient_data", "manual_review_required",
    )
    assert isinstance(j["matched_protocols"], list)
    assert "structural_assessment" in j
    assert "protocol_assessment" in j
    assert "doctor_summary" in j
    assert "send_gate" in j
    assert "gate_allowed" in j["send_gate"]
    assert "sign_decision" in j["send_gate"]


def test_markdown_report_sections():
    doc, rep = _report()
    md = report_to_markdown(rep, doc)
    for header in [
        "# Оценка консультативного заключения",
        "## 1. Краткое резюме",
        "## 2. Проверка структуры КЗ",
        "## 3. Данные пациента",
        "## 4. Проверка диагноза",
        "## 5. Применимость протоколов",
        "## 6. Проверка обследований",
        "## 7. Проверка лечения",
        "## 8. Красные флаги и безопасность",
        "## 9. Повторная явка и контроль",
        "## 10. Все замечания",
        "## 11. Источники",
    ]:
        assert header in md
    assert "gastro/g.pdf" in md
    # ФИО не попадает в отчёт — только инициалы или прочерк.
    assert "Пациент:" in md


def test_html_report_has_structure():
    doc, rep = _report()
    html = report_to_html(rep, doc)
    assert "consult-report-html" in html
    assert "cr-hero-grid" in html
    assert "cr-badge" in html
    assert "<script" not in html.lower()
    assert "cr-sign-decision" in html
    assert "Решение о подписи" in html


def test_html_report_sign_gate_uses_headline_score():
    doc, rep = _report()
    html = report_to_html(rep, doc, headline_score=55.0)
    assert "Нужно подтверждение врача" in html
    assert "Можно подписывать" not in html

    sg = report_to_json(rep, doc, headline_score=55.0)["send_gate"]
    assert sg["sign_decision"] == "review_required"
    assert sg["gate_score"] == 55.0


def test_patch_report_html_send_gate():
    from clinical_knowledge.compliance_gate import evaluate_send_gate
    from clinical_knowledge.consult_report import patch_report_html_send_gate

    doc, rep = _report()
    html = report_to_html(rep, doc)
    assert "Можно подписывать" in html
    sg = evaluate_send_gate(rep, headline_score=58.0, mode="soft_gate", min_score_hard=70.0)
    patched = patch_report_html_send_gate(html, sg)
    assert "Нужно подтверждение врача" in patched
    assert patched.count("cr-sign-decision") == 1

