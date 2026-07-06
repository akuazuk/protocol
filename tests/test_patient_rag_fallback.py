"""Тесты RAG fallback / semantic rescue для B2C patient review."""
from __future__ import annotations

from clinical_knowledge.patient_protocol_retrieval import (
    merge_patient_rag_context,
    rag_probe_looks_like_kz,
)
from clinical_knowledge.patient_rag_questions import augment_questions_from_retrieved
from clinical_knowledge.patient_review import _l1_needs_rag_fallback, run_patient_review

SHORT_KZ = "Диагноз: ОРВИ. Р: симптоматически, парацетамол. Контроль 7 дн."
NOISY_OCR = (
    "Жaлoбы: кашeль 3 дня\n"
    "Диaгноз ОРВИ\n"
    "Рек0мендации: питьё, парацетамол\n"
    "Контроль"
)
GOOD_KZ = (
    "Врач: флеболог\n"
    "Жалобы: отёк правой голени.\n"
    "Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.\n"
    "Рекомендации: ривароксабан 20 мг 1 раз в день.\n"
    "Контроль через 3 месяца."
)
MOCK_RETRIEVED = [
    {
        "text": "При ОРВИ рекомендуется симптоматическое лечение и контроль самочувствия через 5-7 дней.",
        "section_title": "Лечение",
        "path": "minzdrav_protocols/orvi/kp_orvi.pdf",
    },
    {
        "text": "Показан повторный осмотр при сохранении лихорадки более 3 суток или ухудшении.",
        "section_title": "Наблюдение",
        "path": "minzdrav_protocols/orvi/kp_orvi.pdf",
    },
]


def test_l1_needs_rag_fallback_thresholds() -> None:
    assert _l1_needs_rag_fallback({"matched_protocols_count": 0}) is True
    assert _l1_needs_rag_fallback({"matched_protocols_count": 2, "confidence_score": 40}) is True
    assert _l1_needs_rag_fallback({"matched_protocols_count": 3, "confidence_score": 72}) is False


def test_rag_probe_looks_like_kz_conservative() -> None:
    probe_ok = {"retrieved": MOCK_RETRIEVED}
    assert rag_probe_looks_like_kz(probe_ok, SHORT_KZ) is True
    assert rag_probe_looks_like_kz({"retrieved": MOCK_RETRIEVED[:1]}, SHORT_KZ) is False
    assert rag_probe_looks_like_kz(probe_ok, "случайный текст без медицины") is False


def test_merge_patient_rag_context_dedupes() -> None:
    base = {"paths": ["a.pdf"], "retrieved": [MOCK_RETRIEVED[0]], "icd_codes": ["J06"]}
    extra = {
        "paths": ["a.pdf", "b.pdf"],
        "retrieved": MOCK_RETRIEVED,
        "icd_codes": ["J06", "J00"],
    }
    merged = merge_patient_rag_context(base, extra)
    assert merged["paths"] == ["a.pdf", "b.pdf"]
    assert len(merged["retrieved"]) == 2


def test_augment_questions_from_retrieved_adds_plain_context() -> None:
    report = {"questions_structured": [], "questions_for_doctor": [], "action_checklist": []}
    out = augment_questions_from_retrieved(
        report,
        retrieved=MOCK_RETRIEVED,
        kz_text=SHORT_KZ,
        limit=2,
    )
    qs = out.get("questions_structured") or []
    assert len(qs) >= 1
    assert any(q.get("plain_context") for q in qs)
    assert all("?" in q.get("text", "") for q in qs)


def test_semantic_rescue_short_kz(monkeypatch) -> None:
    monkeypatch.setenv("PATIENT_UPLOAD_SEMANTIC_RESCUE", "1")
    monkeypatch.setenv("PATIENT_RAG_RETRIEVAL_ENABLED", "1")

    def fake_retrieve(**kwargs):
        if kwargs.get("semantic_primary"):
            return {
                "paths": ["minzdrav_protocols/orvi/kp_orvi.pdf"],
                "retrieved": MOCK_RETRIEVED,
                "icd_codes": [],
                "rag_used": True,
                "vector_used": False,
                "semantic_primary": True,
            }
        return {
            "paths": ["minzdrav_protocols/orvi/kp_orvi.pdf"],
            "retrieved": MOCK_RETRIEVED,
            "icd_codes": [],
            "rag_used": True,
            "vector_used": False,
            "semantic_primary": False,
        }

    import clinical_knowledge.patient_protocol_retrieval as ppr

    monkeypatch.setattr(ppr, "retrieve_patient_protocol_context", fake_retrieve)

    out = run_patient_review(text=SHORT_KZ, consultation_id="t-rescue")
    assert out.get("upload_mismatch") is not True
    pr_report = out.get("patient_report") or {}
    assert pr_report.get("document_quality_tag") == "short_or_ocr"


def test_good_kz_unchanged_with_fallback_flags(monkeypatch) -> None:
    monkeypatch.setenv("PATIENT_RAG_SEMANTIC_FALLBACK", "1")
    monkeypatch.setenv("PATIENT_RAG_QUESTIONS_ENABLED", "1")
    monkeypatch.setenv("PATIENT_UPLOAD_SEMANTIC_RESCUE", "1")
    out = run_patient_review(text=GOOD_KZ, consultation_id="t-good-regression")
    assert out.get("upload_mismatch") is not True
    assert (out.get("matched_protocols_count") or 0) >= 1
    pr = out.get("patient_report") or {}
    meta = pr.get("protocol_rag_meta") or {}
    assert meta.get("fallback_used") is not True
    assert meta.get("semantic_rescue") is not True
