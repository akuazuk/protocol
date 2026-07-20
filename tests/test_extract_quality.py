"""Контроль качества клинических выдержек (отсев обрывков и мусора)."""
from __future__ import annotations

from clinical_knowledge.extract_quality import (
    best_meaningful_excerpt,
    clean_clinical_text,
    dedupe_meaningful,
    is_legal_admin_text,
    is_meaningful_clinical_text,
    meaningful_clinical_excerpt,
)


def test_single_word_is_not_meaningful() -> None:
    assert is_meaningful_clinical_text("УЗИ") is False
    assert is_meaningful_clinical_text("контроль") is False
    assert is_meaningful_clinical_text("норадреналином") is False


def test_full_sentence_is_meaningful() -> None:
    t = "Рентгенография органов грудной клетки выполняется всем пациентам с подозрением на пневмонию."
    assert is_meaningful_clinical_text(t) is True


def test_enumeration_prefix_stripped() -> None:
    assert clean_clinical_text("4.1. малые критерии тяжелого течения") == "малые критерии тяжелого течения"


def test_dangling_tail_and_unbalanced_paren_trimmed() -> None:
    t = "4.1. малые критерии тяжёлого течения пневмонии (далее - малый"
    out = meaningful_clinical_excerpt(t, limit=240)
    assert "(" not in out
    assert not out.endswith("далее")
    assert out.startswith("малые критерии")


def test_garbled_short_returns_empty() -> None:
    assert meaningful_clinical_excerpt("УЗИ", limit=240) == ""
    assert meaningful_clinical_excerpt("контроль", limit=240) == ""


def test_best_excerpt_prefers_first_meaningful() -> None:
    out = best_meaningful_excerpt(
        ["Амоксициллин", "Амоксициллин назначают внутрь по 500 мг три раза в сутки семь дней."],
        limit=240,
    )
    assert out.startswith("Амоксициллин назначают")


def test_best_excerpt_all_bad_returns_empty() -> None:
    assert best_meaningful_excerpt(["УЗИ", "ЭКГ", None], limit=240) == ""


def test_bracket_tag_prefix_stripped() -> None:
    assert clean_clinical_text("[routing] 6. Направление пациентов в стационар").startswith(
        "Направление пациентов"
    )


def test_boilerplate_rejected() -> None:
    assert is_meaningful_clinical_text("«О здравоохранении», а также термины и их определения") is False
    assert is_meaningful_clinical_text("Настоящий клинический протокол устанавливает требования") is False


def test_glossary_line_rejected() -> None:
    assert is_meaningful_clinical_text("АСИТ - аллергенспецифическая иммунотерапия") is False
    assert is_meaningful_clinical_text("MART-терапия - режим терапии одним ингалятором") is False


def test_icd_shifr_boilerplate_rejected() -> None:
    t = "(шифр по Международной статистической классификации болезней десятого пересмотра - J45)"
    assert is_meaningful_clinical_text(t) is False


def test_dedupe_meaningful_drops_near_duplicates() -> None:
    cands = [
        "Первичную диагностику бронхиальной астмы в амбулаторных условиях осуществляют врачи-терапевты.",
        "5. Первичную диагностику бронхиальной астмы в амбулаторных условиях осуществляют врачи-терапевты.",
        "Степень тяжести течения астмы у пациентов, получающих лечение, определяется объёмом терапии.",
    ]
    out = dedupe_meaningful(cands, limit=240)
    assert len(out) == 2
    assert out[0].startswith("Первичную диагностику")
    assert out[1].startswith("Степень тяжести")


def test_sentence_start_gate_rejects_midword_leads() -> None:
    frag = "ьзование в качестве неотложной медицинской помощи комбинаций ИГКС осуществляется"
    assert is_meaningful_clinical_text(frag) is True
    assert is_meaningful_clinical_text(frag, require_sentence_start=True) is False
    assert meaningful_clinical_excerpt(frag, require_sentence_start=True) == ""


def test_sentence_start_gate_keeps_capitalized() -> None:
    good = "Диагноз астмы устанавливают по клинической картине и обратимости обструкции бронхов."
    assert meaningful_clinical_excerpt(good, require_sentence_start=True) == good


def test_glossary_abbr_definition_rejected() -> None:
    assert is_meaningful_clinical_text(
        "MART-терапия - режим терапии с использованием одного комбинированного ингалятора"
    ) is False
    assert is_meaningful_clinical_text("ACQ тест - опросник по контролю над симптомами астмы") is False
    # содержательное определение основного термина (длинное) не задевается
    assert is_meaningful_clinical_text(
        "БА - гетерогенное заболевание, характеризующееся хроническим воспалением дыхательных путей "
        "с наличием респираторных симптомов и вариабельной бронхиальной обструкцией."
    ) is True


def test_legal_admin_boilerplate_detected() -> None:
    noise = [
        "Признать утратившим силу приказ Министерства здравоохранения Республики Беларусь от 14 февраля 2011 г. № 150.",
        "Настоящее постановление вступает в силу после его официального опубликования.",
        "Министр Д.Л.Пиневич СОГЛАСОВАНО Брестский областной исполнительный комитет.",
        "Национальный правовой Интернет-портал Республики Беларусь, 16.04.2022, 8/37875 3",
        "ГЛАВА 2",
        "Утвердить клинический протокол «Диагностика и лечение пациентов с тромбозом глубоких вен».",
    ]
    for t in noise:
        assert is_legal_admin_text(t) is True, t
        assert is_meaningful_clinical_text(t) is False, t


def test_clinical_text_not_flagged_as_admin() -> None:
    clinical = [
        "УЗДС является основным методом визуализации при подозрении на ТГВ.",
        "Госпитализация с установленным диагнозом ТГВ осуществляется в хирургическое отделение.",
        "Длина флотирующей части тромба более 5 см является критерием эмболоопасности.",
        "МКБ-10: I80, I82",
    ]
    for t in clinical:
        assert is_legal_admin_text(t) is False, t


def test_excerpt_ends_on_sentence_boundary() -> None:
    t = (
        "Диагноз устанавливают на основании клинической картины. "
        "Далее следует множество дополнительных деталей, которые не должны попасть в выдержку "
        "полностью, поскольку превышают лимит символов и продолжаются очень долго без остановки."
    )
    out = meaningful_clinical_excerpt(t, limit=60)
    assert out
    assert len(out) <= 62
