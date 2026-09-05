"""Wave 2: therapeutic class duplicates (shadow except NSAID primary path)."""
from __future__ import annotations

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.medication_safety import (
    all_therapeutic_class_dups,
    concurrent_systemic_class,
    load_therapeutic_classes,
)
from clinical_knowledge.mo_finding_labels_ru import FINDING_TITLE_RU


def test_therapeutic_classes_dictionary_size() -> None:
    classes = load_therapeutic_classes()
    assert len(classes) >= 5
    ids = {c["id"] for c in classes}
    assert {"ppi", "antihistamine", "anticoag_antiplatelet", "statin", "ace_arb"} <= ids


def test_ppi_dup_detected_not_alternatives() -> None:
    text = "Омепразол 20 мг утром и пантопразол 40 мг вечером"
    labels = concurrent_systemic_class(text, "ppi")
    assert len(labels) >= 2
    alt = "Омепразол 20 мг (или пантопразол, или эзомепразол)"
    assert concurrent_systemic_class(alt, "ppi") == []


def test_class_dups_shadow_in_deep_eval(monkeypatch) -> None:
    monkeypatch.delenv("MO_CLASS_DUP_PRIMARY", raising=False)
    case = {
        "complaints": "изжога",
        "anamnesis_doctor": "длительно",
        "objective_status": "живот мягкий",
        "clinical_diagnosis": "ГЭРБ",
        "treatment_recommendations": "Омепразол 20 мг и Нольпаза 40 мг",
        "exam_recommendations": "",
    }
    result = evaluate_kz_deep(case, drug_ctx={})
    shadow_codes = {f["code"] for f in result.get("shadow_findings") or []}
    primary_codes = {f["code"] for f in result.get("findings") or []}
    assert "C_ppi_dup" in shadow_codes
    assert "C_ppi_dup" not in primary_codes


def test_class_dup_labels() -> None:
    assert "ИПП" in FINDING_TITLE_RU["C_ppi_dup"]
    assert FINDING_TITLE_RU["C_statin_dup"]


def test_all_therapeutic_class_dups_helper() -> None:
    hits = all_therapeutic_class_dups(
        "Аторвастатин 20 мг и розувастатин 10 мг"
    )
    assert any(h["class_id"] == "statin" for h in hits)
