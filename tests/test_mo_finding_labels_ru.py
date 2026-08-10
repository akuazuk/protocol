"""Русские названия кодов замечаний МО."""
from __future__ import annotations

from clinical_knowledge.mo_finding_labels_ru import (
    demote_stale_reg55_p0,
    enrich_finding_detail_ru,
    finding_label_ru,
    priority_from_score,
    queue_priority_for_case,
    severity_label_ru,
    severity_tone_css,
    source_ref_display_ru,
)


def test_known_codes_have_russian_titles() -> None:
    assert "обследован" in finding_label_ru("B_exams_gap").lower()
    assert "МКБ" in finding_label_ru("B_icd_invalid")
    assert "наблюден" in finding_label_ru("A_missing_follow_up").lower()
    assert "лечен" in finding_label_ru("A_missing_treatment").lower()
    assert finding_label_ru("B_exams_gap", "B_exams_gap").startswith("Не отражены")


def test_keeps_existing_russian_title() -> None:
    assert finding_label_ru("B_exams_gap", "Уже понятный текст") == "Уже понятный текст"


def test_source_ref_display_ru_for_common_refs() -> None:
    assert "DDInter" in source_ref_display_ru("DDInter")
    assert "взаимодейств" in source_ref_display_ru("DDInter").lower()
    assert "НПВС" in source_ref_display_ru("ISMP/клин.практика")
    assert "21.05.2021" in source_ref_display_ru("Пост. №55")
    text = source_ref_display_ru("template_pair:430cee3c370509317834e5bd14a9c4b3:899506")
    assert "899506" in text
    assert "сходств" in text.lower() or "шаблон" in text.lower()


def test_enrich_replaces_generic_template_why() -> None:
    out = enrich_finding_detail_ru(
        code="E_template_copy",
        detail="Требует проверки, что индивидуальные данные пациента отражены полностью.",
        source_ref="template_pair:abc:12345",
    )
    assert "12345" in out
    assert "требует проверки, что индивидуальные" not in out.lower()


def test_severity_label_ru_plain_words() -> None:
    assert severity_label_ru("P0") == "Критично"
    assert severity_label_ru("P1") == "Важно"
    assert severity_label_ru("P2") == "Умеренно"
    assert severity_label_ru("P3") == "Оформление"
    assert "P0" not in severity_label_ru("P0")
    assert severity_tone_css("P0") == "critical"
    assert severity_tone_css("P1") == "important"
    assert severity_tone_css("P2") == "check"
    assert severity_tone_css("P3") == "formal"


def test_demote_stale_reg55_p0_when_catalog_has_no_p0() -> None:
    out = demote_stale_reg55_p0(
        code="D_reg55_p0",
        severity="P0",
        title_ru="Критический дефект по №55",
    )
    assert out["demoted_stale_reg55_p0"] is True
    assert out["severity"] == "P1"
    assert out["severity_label_ru"] == "Важно"
    assert "критический дефект" not in out["title_ru"].lower()


def test_priority_from_score_bands() -> None:
    assert priority_from_score(35)["label_ru"] == "Критично"
    assert priority_from_score(55)["label_ru"] == "Важно"
    assert priority_from_score(70)["label_ru"] == "Умеренно"
    assert priority_from_score(90)["label_ru"] == "Оформление"


def test_queue_priority_uses_formula_after_demote() -> None:
    prio = queue_priority_for_case(
        finding_severity="P1",
        score_pct=40.0,
        axes={
            "documentation": 80,
            "clinical_concordance": 75,
            "safety": 90,
            "regulatory": 70,
        },
        demoted_stale_reg55_p0=True,
    )
    assert prio["formula_pct"] is not None
    assert float(prio["formula_pct"]) > 40
    assert prio["label_ru"] != "Критично"
