"""Тесты русификации подписей правил и evidence map."""
from __future__ import annotations

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.evidence_map import build_evidence_map
from clinical_knowledge.rule_labels_ru import (
    decision_ru,
    localize_message_ru,
    population_ru,
    rule_title_ru,
)

GASTRO = """\
Врач: гастроэнтеролог
Дата консультации: 14.07.2024
Дата рождения: 12.05.1976
Диагноз: K29.7 Хронический гастрит.
Рекомендации по лечению: Омепразол 20 мг.
"""


def test_rule_title_ru_from_auto_id():
    title = rule_title_ru("9f9e0fb1_auto_gastritis_diagnosis_formula", {})
    assert "гастрит" in title.lower()
    assert "диагноз" in title.lower()
    assert "9f9e0fb1" not in title


def test_rule_title_ru_gerd_diagnosis_formula():
    title = rule_title_ru("9f9e0fb1_auto_gerd_diagnosis_formula", {"rule_type": "diagnosis_formula"})
    assert "ГЭРБ" in title
    assert "диагноз" in title.lower()


def test_decision_ru():
    assert decision_ru("missing") == "Не выполнено"
    assert decision_ru("satisfied") == "Выполнено"


def test_population_ru_in_message():
    msg = localize_message_ru("Протокол для child, в КЗ аудитория adult.")
    assert "child" not in msg.lower()
    assert "adult" not in msg.lower()
    assert population_ru("child") == "дети"
    assert population_ru("adult") == "взрослые"


def test_evidence_map_uses_russian_labels():
    doc = parse_consultation(GASTRO, consultation_id="t")
    rules_check = {
        "findings": [
            {
                "rule_id": "9f9e0fb1_auto_gastritis_diagnosis_formula",
                "rule_type": "diagnosis_formula",
                "passed": False,
                "message_ru": "В формулировке диагноза не хватает компонентов: этиология",
                "required_components": ["этиология"],
            },
            {
                "rule_id": "celiac_population_guard",
                "rule_type": "population_mismatch",
                "passed": False,
                "message_ru": "Нозология «Целиакия» - протокол для дети, в КЗ аудитория взрослые.",
            },
        ]
    }
    items = build_evidence_map(doc, rules_check)
    assert len(items) == 1
    item = items[0]
    assert item.title_ru
    assert "9f9e0fb1" not in item.title_ru
    assert item.decision_ru == "Не выполнено"
    assert item.explanation
    assert "этиолог" in item.explanation.lower()
