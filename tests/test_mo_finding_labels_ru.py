"""Русские названия кодов замечаний МО."""
from __future__ import annotations

from clinical_knowledge.mo_finding_labels_ru import (
    enrich_finding_detail_ru,
    finding_label_ru,
    severity_label_ru,
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


def test_severity_label_ru() -> None:
    assert severity_label_ru("P0").startswith("P0")
    assert "критич" in severity_label_ru("P0").lower()
