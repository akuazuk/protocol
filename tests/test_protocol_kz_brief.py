"""Тесты KZ-brief и structured excerpt."""
from __future__ import annotations

from clinical_knowledge.protocol_kz_brief import (
    build_kz_brief,
    clear_protocol_brief_cache,
    kz_brief_to_heuristic_matrix,
    resolve_protocol_brief_bundle_cached,
)
from clinical_knowledge.protocol_summary.schema import (
    ConditionSummary,
    CriteriaBlock,
    CriterionItem,
    DrugTreatmentItem,
    ExamRequirement,
    ProtocolSource,
    ProtocolSummary,
    SummarySourceRef,
    TreatmentBlock,
)
from clinical_knowledge.structured_retrieval_excerpt import (
    attach_structured_excerpts,
    build_structured_excerpt_for_path,
)


def _src() -> SummarySourceRef:
    return SummarySourceRef(page_start=12, section_title="3.2 Лечение", quote="Цитата из протокола.")


def _sample_summary() -> ProtocolSummary:
    ref = _src()
    cond = ConditionSummary(
        condition_id="j209",
        name="Острый бронхит",
        icd10_codes=["J20.9"],
        diagnostic_criteria=CriteriaBlock(
            required=[
                CriterionItem(text="Кашель менее 3 недель без признаков пневмонии", source_ref=ref),
            ],
        ),
        required_exams=[
            ExamRequirement(
                name="ОАК",
                requirement_level="required",
                source_ref=ref,
            ),
        ],
        treatment=TreatmentBlock(
            drugs=[
                DrugTreatmentItem(
                    drug_name="Амоксициллин",
                    dose_text="500 мг",
                    frequency_text="3 р/сут",
                    duration_text="5-7 дней",
                    source_ref=ref,
                ),
            ],
        ),
    )
    return ProtocolSummary(
        protocol_id="test_proto",
        source=ProtocolSource(title="КП тест", local_path="minzdrav/test/bronhit.pdf"),
        conditions=[cond],
    )


def test_build_kz_brief_from_summary(monkeypatch):
    summary = _sample_summary()

    def fake_find(path: str):
        return summary if "bronhit" in path else None

    monkeypatch.setattr(
        "clinical_knowledge.protocol_summary.nav.find_summary_by_catalog_path",
        fake_find,
    )
    out = build_kz_brief(
        "minzdrav/test/bronhit.pdf",
        condition_id="j209",
        icd_codes=["J20.9"],
    )
    assert out["available"] is True
    assert out["coverage"]["blocks_filled"] >= 3
    texts = " ".join(s["text"] for sec in out["sections"] for s in (sec.get("items") or []))
    assert "Амоксициллин" in texts


def test_brief_bundle_cache(monkeypatch):
    clear_protocol_brief_cache()
    summary = _sample_summary()

    def fake_find(path: str):
        return summary

    monkeypatch.setattr(
        "clinical_knowledge.protocol_summary.nav.find_summary_by_catalog_path",
        fake_find,
    )
    path = "minzdrav/test/bronhit.pdf"
    r1 = resolve_protocol_brief_bundle_cached(path, condition_id="j209", query="кашель")
    r2 = resolve_protocol_brief_bundle_cached(path, condition_id="j209", query="кашель")
    assert r1.get("cache_hit") is False
    assert r2.get("cache_hit") is True
    assert r1["kz_matrix"]["sections"]


def test_kz_brief_heuristic_matrix():
    kz = {
        "summary_ru": "тест",
        "sections": [
            {
                "kz_section": "Обследование",
                "items": [{"text": "ОАК", "obligation": "required", "quote": "ОАК обязателен"}],
            },
        ],
    }
    matrix = kz_brief_to_heuristic_matrix(
        kz,
        path="minzdrav/x.pdf",
        title="КП",
        icd_codes=["J20.9"],
    )
    assert matrix["sections"][0]["items"][0]["text"] == "ОАК"
    assert len(matrix["sections"][0]["items"][0]["protocol_excerpt"]) > 10


def test_structured_excerpt_groups_by_kind():
    retrieval = [
        {"path": "minzdrav/a.pdf", "kind": "diagnostics", "excerpt": "Назначить ОАК и биохимию.", "score": 0.9},
        {"path": "minzdrav/a.pdf", "kind": "pharmacotherapy", "excerpt": "Амоксициллин 500 мг 3 раза в сутки.", "score": 0.85},
        {"path": "minzdrav/b.pdf", "kind": "body", "excerpt": "Другой протокол.", "score": 0.5},
    ]
    out = build_structured_excerpt_for_path(retrieval, "minzdrav/a.pdf")
    assert out["available"] is True
    assert len(out["sections"]) >= 2
    assert "diagnostics" in out["snippets"]


def test_attach_structured_excerpts():
    payload = {"llm_json": {"protocols": [{"path": "minzdrav/a.pdf"}, {"path": "minzdrav/b.pdf"}]}}
    retrieval = [
        {"path": "minzdrav/a.pdf", "kind": "diagnostics", "excerpt": "ОАК.", "score": 0.9},
    ]
    attach_structured_excerpts(payload, retrieval, limit=2)
    assert "protocol_excerpts" in payload
    assert payload["protocol_excerpts"]["minzdrav/a.pdf"]["available"] is True
