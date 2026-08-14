from __future__ import annotations

from clinical_knowledge.protocol_content_index import (
    clear_content_index_cache,
    content_text_for_card,
    content_text_for_path,
)


def test_content_index_has_hemorrhoid_in_rectal_protocol() -> None:
    path = (
        "minzdrav_protocols/khirurgiya/"
        "КП_Диагностика_лечение_пациентов_взрос_с_доброкач_забол_"
        "прямой_кишки_параректальной_и_копчиковой_области_амбул_пост_МЗ_01.04.2022_№22.pdf"
    )
    clear_content_index_cache()
    text = content_text_for_path(path).lower()
    assert "геморро" in text
    assert "прямой" in text
    via_card = content_text_for_card({"source_path": path}).lower()
    assert "геморро" in via_card


def test_hemorrhoid_alias_does_not_match_hemorrhagic_disease() -> None:
    from clinical_knowledge.dx_query_expand import expand_diagnosis_query, matched_alias_phrases

    assert matched_alias_phrases("геморрагическая болезнь новорожденных") == []
    expanded = expand_diagnosis_query("Геморрой неуточненный").lower()
    assert "геморроидальный" in expanded
    assert "прямой кишки" in expanded
