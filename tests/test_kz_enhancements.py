"""Тесты дополнений KZ: rich rules, typed retrieve, index, sync."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.consult_alignment import (
    append_alignment_evidence,
    sync_structured_with_alignment,
)
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.consult_retrieval import (
    supplement_retrieval_from_rich_chunks,
    unify_consult_protocol_paths,
)
from clinical_knowledge.dispensary_regulations import lookup_follow_up_expectations
from clinical_knowledge.rich_rules_supplement import rich_table_rules_for_paths

FIXTURE_CHUNKS = Path(__file__).resolve().parent / "fixtures" / "chunks.mini.jsonl"


def test_unify_consult_protocol_paths():
    out = unify_consult_protocol_paths(
        target_paths=["a.pdf", "b.pdf"],
        rules_paths=["b.pdf", "c.pdf"],
        rag_paths=["d.pdf"],
        max_paths=5,
    )
    assert out == ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]


def test_supplement_retrieval_from_rich_chunks():
    def get_chunks(path: str):
        if "x" in path:
            return [{
                "chunk_type": "diagnostics",
                "text": "УЗИ вен нижних конечностей обязательное исследование при тромбозе.",
                "page_from": 5,
                "section_title": "Диагностика",
            }]
        return []

    out = supplement_retrieval_from_rich_chunks(
        [],
        paths=["minz/x.pdf"],
        icd_codes=["I80.1"],
        get_chunks=get_chunks,
    )
    assert len(out) >= 1
    assert out[0].get("typed_retrieve") is True


def test_rich_table_rules_for_paths(monkeypatch):
    if not FIXTURE_CHUNKS.is_file():
        return
    monkeypatch.setenv("RAG_CHUNKS_JSONL", str(FIXTURE_CHUNKS))
    from clinical_knowledge.rich_rules_supplement import _chunks_index_cached

    _chunks_index_cached.cache_clear()
    rules = rich_table_rules_for_paths(["any/path.pdf"])
    assert isinstance(rules, list)


def test_sync_structured_with_alignment():
    structured = {"compliance": {"score_breakdown": {"diagnosis_score": 90}}}
    alignment = {
        "alignment_cards": [{
            "block_id": "diagnosis",
            "name_ru": "Диагноз",
            "score_pct": 95,
            "source_kind": "mkb",
            "source_label": "МКБ-10",
        }],
    }
    sync_structured_with_alignment(structured, alignment)
    assert structured["compliance"]["alignment_by_block"]["diagnosis"]["alignment_score"] == 95


def test_append_alignment_evidence():
    structured = {"compliance": {"evidence_map": []}}
    alignment = {
        "alignment_cards": [{
            "block_id": "exams",
            "name_ru": "Обследование",
            "score_pct": 80,
            "protocol_excerpt": "УЗИ",
            "conclusion_excerpt": "назначено",
            "source_kind": "kp",
        }],
    }
    append_alignment_evidence(structured, alignment)
    assert len(structured["compliance"]["evidence_map"]) == 1
    assert structured["compliance"]["evidence_map"][0]["rule_source"] == "alignment"


def test_lookup_i80_prefix():
    reg = lookup_follow_up_expectations(["I80.1"])
    hints = " ".join(reg.get("follow_up_hints") or [])
    assert "I80" in reg.get("icd_chapters") or "I" in reg.get("icd_chapters")
    assert hints


def test_l1_includes_alignment():
    from clinical_knowledge.consult_tiering import run_l1_structured_review

    text = """\
Врач: терапевт
Жалобы: кашель.
Объективный статус: удовлетворительное.
Диагноз: J06.8 Острая инфекция.
Контроль через неделю.
"""
    out = run_l1_structured_review(text=text, consultation_id="l1_test")
    assert out.get("review_tier") == "L1"
    assert out.get("criteria_source") == "deterministic_alignment" or out.get("review", {}).get("criteria")
