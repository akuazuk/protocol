"""L2-fast: детерминированный review, evidence pack, конфиг."""
from __future__ import annotations

import os

from clinical_knowledge.consult_evidence_pack import build_evidence_pack
from clinical_knowledge.consult_l2_config import (
    consult_l2_fast_enabled,
    resolve_l2_mode,
)
from clinical_knowledge.consult_l2_review import (
    build_l2_fast_review,
    extract_block_gaps,
    template_summary_ru,
)


def test_consult_l2_fast_enabled_default_on_lite(monkeypatch) -> None:
    monkeypatch.delenv("CONSULT_L2_FAST", raising=False)
    monkeypatch.setenv("CONSULT_RENDER_L2_LITE", "1")
    assert consult_l2_fast_enabled() is True


def test_resolve_l2_mode_fast_vs_narrative() -> None:
    assert resolve_l2_mode(narrative=False) in ("fast", "evidence")
    assert resolve_l2_mode(narrative=True) == "narrative"


def test_template_summary_from_critical_issues() -> None:
    sa = {
        "compliance": {
            "critical_issues": [
                {"message_ru": "Нет кода МКБ"},
                {"message_ru": "Слабое описание лечения"},
            ]
        }
    }
    txt = template_summary_ru(sa)
    assert "МКБ" in txt
    assert "лечения" in txt


def test_build_l2_fast_review_has_summary() -> None:
    sa = {"compliance": {"overall_score": 72, "overall_status": "ok"}}
    align = {"limitations_ru": "Тест ограничение"}
    rev = build_l2_fast_review(structured_analysis=sa, alignment_result=align)
    assert rev["criteria_source"] == "deterministic_l2_fast"
    assert rev["summary_ru"]
    assert rev["limitations_ru"] == "Тест ограничение"


def test_extract_block_gaps_from_alignment_cards() -> None:
    align = {
        "alignment_cards": [
            {
                "block_id": "treatment",
                "name_ru": "Лечение",
                "gaps_ru": ["Нет дозировки"],
                "score_pct": 40,
                "comment_ru": "Слабое соответствие",
            }
        ]
    }
    gaps = extract_block_gaps(align)
    assert any("дозировки" in g["gap_ru"] for g in gaps)


def test_build_evidence_pack_empty_paths() -> None:
    pack = build_evidence_pack(
        icd_codes=["J06.9"],
        match_paths=[],
        get_chunks=lambda _p: [],
    )
    assert "blocks" in pack
    assert isinstance(pack.get("fragment_count"), int)


def test_evidence_pack_to_protocol_rows() -> None:
    from clinical_knowledge.consult_evidence_pack import evidence_pack_to_protocol_rows

    pack = {
        "blocks": {
            "exams": [
                {
                    "protocol_path": "minzdrav_protocols/a.pdf",
                    "excerpt": "Рентгенография органов грудной клетки при подозрении на пневмонию.",
                    "section": "Диагностика",
                    "page": 12,
                    "block_id": "exams",
                }
            ]
        }
    }
    rows = evidence_pack_to_protocol_rows(pack)
    assert len(rows) == 1
    assert rows[0]["path"] == "minzdrav_protocols/a.pdf"


def test_make_chunk_cache_dedupes_reads() -> None:
    from clinical_knowledge.consult_memory import make_chunk_cache

    calls = {"n": 0}

    def _src(path: str) -> list[dict]:
        calls["n"] += 1
        return [{"text": f"chunk-{path}", "path": path}]

    cached = make_chunk_cache(_src)
    assert len(cached("p1.pdf")) == 1
    assert len(cached("p1.pdf")) == 1
    assert calls["n"] == 1


def test_consult_l2_skip_rag_warm_manifest(monkeypatch) -> None:
    import rag_server as rs

    monkeypatch.setenv("CONSULT_L2_FAST", "1")
    monkeypatch.setenv("CONSULT_RENDER_L2_LITE", "1")
    monkeypatch.setenv("RAG_STARTUP_MODE", "manifest")
    assert rs._consult_l2_skip_rag_warm() is True


def test_l2_fast_skips_synthesize(monkeypatch) -> None:
    import consult_review_pipeline as crp
    import rag_server as rs

    monkeypatch.setenv("CONSULT_L2_FAST", "1")
    monkeypatch.setenv("CONSULT_RENDER_L2_LITE", "1")
    rs._consult_review_cache.clear()

    synthesize_called = {"n": 0}

    def _boom(*args, **kwargs):
        synthesize_called["n"] += 1
        raise AssertionError("synthesize should not run in L2-fast")

    monkeypatch.setattr(rs, "_consult_review_synthesize", _boom)
    monkeypatch.setattr(
        rs,
        "get_rich_chunks_for_consult",
        lambda _p: [
            {
                "text": "Диагностика: рентгенография органов грудной клетки при подозрении на пневмонию.",
                "kind": "diagnostics",
                "section_title": "Диагностика",
            }
        ],
    )

    text = (os.path.join(os.path.dirname(__file__), "fixtures/consultations/feedback_urvi_j06.txt"))
    with open(text, encoding="utf-8") as f:
        kz = f.read()

    out = None
    for kind, payload in crp.iter_consult_review_pipeline(
        full_text=kz,
        n_files=1,
        consult_docs_meta=[{"filename": "t.txt"}],
        pdf_warnings=[],
        content_signature="l2fast-test",
        category_slugs="",
    ):
        if kind == "done":
            out = payload

    assert out is not None
    assert out.get("l2_mode") == "fast"
    assert out.get("evidence_pack") is not None
    assert synthesize_called["n"] == 0
    assert out.get("review", {}).get("criteria_source") in (
        "deterministic_l2_fast",
        "deterministic_alignment",
    )
