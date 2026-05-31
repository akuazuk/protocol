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


def test_markdown_report_sections():
    doc, rep = _report()
    md = report_to_markdown(rep, doc)
    for header in [
        "# Оценка консультативного заключения",
        "## 1. Краткое резюме",
        "## 2. Применимость протокола",
        "## 3. Оценка диагноза",
        "## 4. Оценка обследований",
        "## 5. Оценка лечения",
        "## 6. Красные флаги и безопасность",
        "## 7. Качество оформления КЗ",
        "## 8. Ссылки на источники",
    ]:
        assert header in md
    assert "gastro/g.pdf" in md
    # ФИО не попадает в отчёт — только инициалы или прочерк.
    assert "Пациент:" in md


def test_html_report_has_structure():
    doc, rep = _report()
    html = report_to_html(rep, doc)
    assert "consult-report-html" in html
    assert "cr-bar-fill" in html
    assert "cr-badge" in html
    assert "<script" not in html.lower()

