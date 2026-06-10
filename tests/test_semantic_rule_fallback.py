"""Семантический fallback для правил."""
from __future__ import annotations

from clinical_knowledge.semantic_rule_fallback import fuzzy_term_in_text, semantic_presence_check


def test_fuzzy_uzi_synonym():
    text = "Проведено ультразвуковое исследование вен нижних конечностей."
    ok, conf, hint = fuzzy_term_in_text(text, "УЗИ вен")
    assert ok is True
    assert conf >= 0.75


def test_semantic_presence_alias():
    text = "Назначена эхокардиография для оценки функции сердца."
    out = semantic_presence_check(text, "ЭХО-КГ")
    assert out["matched"] is True
    assert out["method"] in ("alias", "fuzzy_tokens", "substring")
