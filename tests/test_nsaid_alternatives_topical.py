"""Gold-driven NSAID rules: скобки-альтернативы и oral+гель."""
from __future__ import annotations

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.medication_safety import (
    concurrent_systemic_nsaids,
    nsaid_labels_in_text,
)
from clinical_knowledge.mo_icd_resolve import resolve_icd_codes_from_mo


def test_nsaid_alternatives_in_parens_not_concurrent():
    text = (
        'При болях "Ибупрофен" ("Кетопрофен", "Найз" и т.п.) в возрастной дозировке'
    )
    assert concurrent_systemic_nsaids(text) == []
    labels = nsaid_labels_in_text(text)
    assert "ибупрофен" in labels
    assert "кетопрофен" not in labels
    assert "нимесулид" not in labels


def test_oral_plus_topical_gel_not_nsaid_dup():
    text = (
        "-аэртал 100 мг 2 р/д №5 дней\n"
        "-местно в эпицентр боли вольтарен- гель смазывать 3 р/д до 7 дней"
    )
    assert concurrent_systemic_nsaids(text) == []
    case = {
        "treatment_recommendations": text,
        "clinical_diagnosis": "Вертеброгенная цервикалгия",
        "complaints": "боль",
        "fields_present": {"diagnosis": True, "complaints": True, "anamnesis": True},
    }
    deep = evaluate_kz_deep(case, protocol_ctx=None, drug_ctx={})
    codes = {f.get("code") for f in deep.get("findings") or []}
    assert "C_nsaid_dup" not in codes


def test_two_systemic_nsaids_still_flagged():
    text = "Ибупрофен 400 мг 3 р/д. Дополнительно мелоксикам 15 мг 1 р/д."
    assert set(concurrent_systemic_nsaids(text)) >= {"ибупрофен", "мелоксикам"}


def test_mis_diagnos_counts_as_icd_present():
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": "Вертеброгенная цервико-торакоалгия",
            "mis_diagnos": "M54.8",
        }
    )
    assert resolved["present"] is True
    assert resolved["main"] == "M54.8"
    assert "M54.8" in resolved["all"]


def test_uro_case_like_no_nsaid_dup_in_deep():
    treatment = (
        "- При болях \"Ибупрофен\" (\"Кетопрофен\", \"Найз\" и т.п.) в возрастной дозировке\n"
        "- Местное лечение: Левомеколь\n"
        "свечи Дикловит № 10 по 1 свече на ночь"
    )
    assert concurrent_systemic_nsaids(treatment) == []
    deep = evaluate_kz_deep(
        {
            "treatment_recommendations": treatment,
            "clinical_diagnosis": "Состояние после циркумцизио",
            "complaints": "боли в области раны",
            "fields_present": {"diagnosis": True, "complaints": True, "objective_status": True},
        },
        protocol_ctx=None,
        drug_ctx={},
    )
    assert "C_nsaid_dup" not in {f.get("code") for f in deep.get("findings") or []}
