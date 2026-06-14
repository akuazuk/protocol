"""ICD-first подбор PDF для воронки поиска протоколов."""
from __future__ import annotations

from clinical_knowledge.search_golden_eval import load_search_golden
from clinical_knowledge.search_retrieval import (
    build_protocol_search_context,
    expand_icd_for_protocol_search,
    search_target_protocol_paths,
)
from rag_server import _rerank_protocols_symptom_only


def test_expand_r07_to_disease_codes_with_throat_text():
    codes, meta = expand_icd_for_protocol_search(
        "болит горло и трудно глотать",
        ["R07.0"],
    )
    assert "R07.0" in codes
    assert any(c.startswith("J02") or c.startswith("J06") for c in codes)
    assert meta.get("had_symptom_only") is True


def test_throat_adult_query_targets_ent_not_anesthesia():
    q = (
        "болит горло и трудно глотать\n"
        "Контекст подбора: взрослое население\n"
        "МКБ-10: R07.0"
    )
    paths, meta = search_target_protocol_paths(query=q, icd_codes=["R07.0"])
    assert meta.get("throat_context") is True
    assert meta.get("strict") is True
    assert paths
    joined = " ".join(paths).lower()
    assert "оторин" in joined or "лор" in joined or "горла" in joined
    assert "анестез" not in joined
    assert "дет_нас" not in joined and "дет нас" not in joined


def test_build_context_sets_allowlist_for_symptom_icd():
    q = "болит горло\nКонтекст подбора: взрослое население\nМКБ-10: R07.0"
    ctx = build_protocol_search_context(query=q, icd_codes=["R07.0"])
    assert ctx.get("path_allowlist")
    expanded = ctx.get("expanded_icd_codes") or []
    assert any(c.startswith("J") for c in expanded)


def test_sym11_throat_r07_golden_row():
    rows = load_search_golden()
    row = next((r for r in rows if r.get("id") == "sym11"), None)
    assert row is not None
    from clinical_knowledge.search_retrieval import build_protocol_search_context

    ctx = build_protocol_search_context(query=row["query"], icd_codes=row.get("icd_codes"))
    assert ctx.get("path_allowlist")
    expanded = ctx.get("expanded_icd_codes") or []
    assert any(c.startswith("J") for c in expanded)


def test_rerank_demotes_anesthesia_and_gi_for_throat():
    protos = [
        {
            "path": "a/anest.pdf",
            "title": "Анестезиологическое обеспечение хирургических вмешательств",
            "confidence_score": 0.98,
        },
        {
            "path": "b/gi.pdf",
            "title": "Диагностика лечение заболеваниями пищевода желудка",
            "confidence_score": 0.95,
        },
        {
            "path": "c/ent.pdf",
            "title": "Диагностика лечение оториноларингологическими заболеваниями в-нас",
            "confidence_score": 0.72,
        },
    ]
    icd = {
        "codes_for_retrieval": ["R07.0"],
        "detected": [{"code": "R07.0"}],
        "suggested": [],
    }
    q = "болит горло и трудно глотать\nКонтекст подбора: взрослое население"
    out = _rerank_protocols_symptom_only(protos, q, icd)
    assert out[0]["path"].endswith("ent.pdf")
