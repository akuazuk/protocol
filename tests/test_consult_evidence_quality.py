"""Фильтры evidence pack и block gaps."""
from __future__ import annotations

from clinical_knowledge.consult_evidence_quality import (
    clean_clinical_sentences,
    is_kp_checklist_item,
    is_reference_noise,
    is_usable_evidence_excerpt,
    is_usable_summary_excerpt,
    normalize_gap_text,
)
from clinical_knowledge.consult_l2_review import extract_block_gaps


def test_normalize_gap_text_strips_bullet() -> None:
    assert normalize_gap_text(" - МРТ грудной клетки") == "МРТ грудной клетки"


def test_is_kp_checklist_item_rejects_toc() -> None:
    assert not is_kp_checklist_item(" - клинический протокол диагностики и лечения инфаркта")
    assert is_kp_checklist_item("МРТ органов грудной клетки")


def test_is_usable_evidence_excerpt_rejects_months() -> None:
    assert not is_usable_evidence_excerpt("июня; октября")


def test_evidence_excerpt_rejects_org_routing_and_admin() -> None:
    # Организационно-маршрутный текст - не клиническая выдержка.
    assert not is_usable_evidence_excerpt(
        "на консультацию к врачу-ангиохирургу (по медицинским показаниям)."
    )
    assert not is_usable_evidence_excerpt(
        "Порядок направления пациентов с ХЗВ определяется Министерством здравоохранения."
    )
    assert not is_usable_evidence_excerpt(
        "устанавливает общие требования к объему оказания медицинской помощи"
    )
    # Обрывок с обрезанным словом.
    assert not is_usable_evidence_excerpt("динамическое наблюден")


def test_summary_excerpt_keeps_short_clinical_names() -> None:
    # Короткие клинические названия обследований должны проходить.
    assert is_usable_summary_excerpt("МРТ")
    assert is_usable_summary_excerpt("УЗДС вен нижних конечностей")
    assert is_usable_summary_excerpt("ОАК")
    # Но мусор так же отсекается.
    assert not is_usable_summary_excerpt("динамическое наблюден")
    assert not is_usable_summary_excerpt(
        "на консультацию к врачу-ангиохирургу (по медицинским показаниям)."
    )


def test_reference_noise_detects_admin_and_normative() -> None:
    assert is_reference_noise("Национальный правовой Интернет-портал Республики Беларусь")
    assert is_reference_noise("06.06.2017 № 59 (стр. 2)")
    assert is_reference_noise("Настоящий клинический протокол устанавливает общие требования")
    assert is_reference_noise("Гемостатики (Группа № 10)")
    assert is_reference_noise("[treatment] что-то там")
    # Клинические названия/фразы - НЕ шум.
    assert not is_reference_noise("МРТ органов малого таза")
    assert not is_reference_noise("Антикоагулянтная терапия низкомолекулярными гепаринами")


def test_empty_pharma_form_rejected() -> None:
    # Только слова лекформ без названия ЛС - пустой фрагмент.
    assert not is_usable_summary_excerpt("таблетки; введения")
    assert not is_usable_summary_excerpt("капсулы")
    # С названием вещества - проходит.
    assert is_usable_summary_excerpt("Эноксапарин, раствор для инъекций")


def test_clean_clinical_sentences_drops_midsentence_and_noise() -> None:
    # Обрезок середины (строчный служебный старт) - отбрасывается.
    assert clean_clinical_sentences("с указанием лекарственной формы и дозировки") is None
    assert clean_clinical_sentences("при объеме кровопотери более 15 % ОЦК выделяются") is None
    # Нормативно-процессное - отбрасывается.
    assert clean_clinical_sentences("Лечение осуществляется в стационарных условиях") is None
    # Валидная клиническая фраза (с содержательного слова / Заглавной) - сохраняется.
    assert clean_clinical_sentences("определение количества плодов при УЗИ") is not None
    res = clean_clinical_sentences("[classification] 4. При постановке диагноза применяется классификация СЕАР (стр. 4)")
    assert res and res.startswith("При постановке диагноза")


def test_extract_block_gaps_collapses_exams_bullets() -> None:
    align = {
        "alignment_cards": [
            {
                "block_id": "exams",
                "name_ru": "Обследование",
                "gaps_ru": [
                    " - клинический протокол диагностики",
                    " - МРТ",
                ],
                "comment_ru": "КП «Тромбоз»: в КЗ отражено 0 из 12 рекомендуемых обследований.",
                "score_pct": 20,
            }
        ]
    }
    gaps = extract_block_gaps(align)
    assert len(gaps) == 1
    assert "0 из 12" in gaps[0]["gap_ru"]
