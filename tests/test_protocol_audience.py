"""Аудитория протоколов, названия и фильтр синтетических выдержек."""
from __future__ import annotations

from clinical_knowledge.protocol_audience import (
    expand_protocol_title_abbreviations,
    infer_protocol_audience,
    is_synthetic_summary_excerpt,
)
from clinical_knowledge.protocol_links import beautify_protocol_title
from rag_server import (
    _filter_protocols_by_funnel_audience,
    _rerank_protocols_symptom_only,
    doc_audience_hint,
    format_excerpt_for_display,
)


def test_infer_protocol_audience_from_filename():
    ped = (
        "minzdrav_protocols/dermatovenerologiya/"
        "КП_Диагностика_лечение_пациентов_детс_нас_папулосквамозными_нарушениями»_пост_МЗ_2024_107.pdf"
    )
    adult = (
        "minzdrav_protocols/dermatovenerologiya/"
        "КП_Диагностика_и_лечение_пациентов_взр.население_с_болезнями_придатков_кожи_постановление_МЗ_2022_59.pdf"
    )
    assert infer_protocol_audience(ped, ped.split("/")[-1]) == "pediatric"
    assert infer_protocol_audience(adult, adult.split("/")[-1]) == "adult"


def test_beautify_expands_population_abbreviations():
    raw = "КП_Диагностика_лечение_пациентов_детс_нас_папулосквамозными_нарушениями»_пост_МЗ_2024_107.pdf"
    title = beautify_protocol_title(raw)
    assert "детское население" in title.lower()
    assert "»" not in title
    assert "_" not in title


def test_synthetic_summary_excerpt_filtered():
    synth = (
        "Протокол: КП детс нас. Нозология: клинический протокол. "
        "МКБ-10: L60.3. Рубрика: Дерматовенерология."
    )
    assert is_synthetic_summary_excerpt(synth)
    assert format_excerpt_for_display(synth, 400) == ""


def test_adult_ingrown_nail_demotes_pediatric_papulosquamous():
    ped_path = (
        "minzdrav_protocols/dermatovenerologiya/"
        "КП_Диагностика_лечение_пациентов_детс_нас_папулосквамозными_нарушениями»_пост_МЗ_2024_107.pdf"
    )
    adult_appendage = (
        "minzdrav_protocols/dermatovenerologiya/"
        "КП_Диагностика_и_лечение_пациентов_взр.население_с_болезнями_придатков_кожи_постановление_МЗ_2022_59.pdf"
    )
    protos = [
        {"path": ped_path, "title": ped_path.split("/")[-1], "confidence_score": 0.98},
        {"path": adult_appendage, "title": adult_appendage.split("/")[-1], "confidence_score": 0.72},
    ]
    q = (
        "вросший ноготь на пальце ноги режет и болит\n"
        "Контекст подбора: взрослое население\n"
        "МКБ-10: L60.0"
    )
    icd = {
        "explicit_icd_in_query": True,
        "detected": [{"code": "L60.0", "title_ru": "Вросший ноготь"}],
        "suggested": [],
        "codes_for_retrieval": ["L60.0"],
    }
    out = _rerank_protocols_symptom_only(protos, q, icd)
    assert out
    assert out[0]["path"] == adult_appendage
    assert doc_audience_hint(ped_path, ped_path.split("/")[-1], {}) == "pediatric"
    assert _filter_protocols_by_funnel_audience([protos[0]], q) == []


def test_expand_protocol_title_abbreviations():
    assert "взрослое население" in expand_protocol_title_abbreviations("взр.население с болезнями")
