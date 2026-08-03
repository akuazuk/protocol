"""Тесты обогащения criteria таблицы КЗ."""
from __future__ import annotations

from clinical_knowledge.consult_alignment import build_consult_alignment
from clinical_knowledge.consult_criteria_enrichment import (
    comment_from_findings_gaps,
    coverage_with_evidence,
    expand_kz_blob,
    kz_evidence_snippet,
    score_from_sop_findings_gaps,
    verify_protocol_excerpt,
)
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.kz_chunk_match import match_kp_item_to_kz


PL_1 = """\
Врач: флеболог
Дата консультации: 12.04.2024
Пол: женский
Жалобы: отёк левой ноги.
Объективный статус: отёк левой ноги.
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации по лечению: ривароксабан 20 мг 1 раз в день.
Рекомендации по обследованию: контрольное УЗИ вен через 3 месяца.
Дата повторной явки: 12.07.2024.
"""


def _fake_chunks(path: str) -> list[dict]:
    if "flebo" in path.lower() or "тромб" in path.lower():
        return [
            {
                "chunk_id": "diag-1",
                "chunk_type": "diagnostics",
                "text": "УЗИ вен нижних конечностей - обязательное исследование при флеботромбозе.",
                "section_title": "Диагностика",
                "page_from": 12,
                "icd10_codes": ["I80.1"],
            },
            {
                "chunk_id": "rx-1",
                "chunk_type": "pharmacotherapy",
                "text": "Ривароксабан 20 мг 1 раз в сутки - прямые оральные антикоагулянты.",
                "section_title": "Лечение",
                "page_from": 24,
                "icd10_codes": ["I80.1"],
            },
        ]
    return []


def test_kz_evidence_snippet_finds_match():
    blob = "Назначено: контрольное УЗИ вен нижних конечностей через 3 месяца."
    sn = kz_evidence_snippet(blob, "УЗИ вен нижних конечностей")
    assert "УЗИ" in sn


def test_match_kp_item_with_raw_text():
    raw = "Полный текст заключения: назначен ривароксабан 20 мг 1 раз в день."
    m = match_kp_item_to_kz("Ривароксабан 20 мг", "", raw_text=raw)
    assert m["kz_match"] in ("found", "partial")


def test_coverage_with_evidence_details():
    meta = [{"text": "УЗИ вен", "obligation": "required"}]
    pct, found, missing, details = coverage_with_evidence(
        ["УЗИ вен нижних конечностей"],
        "Назначено УЗИ вен нижних конечностей",
        meta=meta,
    )
    assert pct >= 80
    assert found
    assert details[0].get("confidence_label")
    assert details[0].get("kz_snippet")


def test_score_from_sop_gaps():
    assert score_from_sop_findings_gaps(["a", "b"], ["c"], base=88) < 88
    assert score_from_sop_findings_gaps(["a"], [], base=50) > 50


def test_comment_from_findings_gaps():
    c = comment_from_findings_gaps(
        "complaints",
        ["Жалобы описаны развёрнуто."],
        ["В жалобах нет давности."],
    )
    assert "Дополните жалобы" in c or "давности" in c


def test_verify_protocol_excerpt_rejects_noise():
    bad = "клинический протокол диагностики и лечения"
    out = verify_protocol_excerpt(bad, cite={"path": "kp.pdf", "page_from": 5})
    assert "Цитата недоступна" in out


def test_alignment_exams_item_details_and_gap_refs():
    doc = parse_consultation(PL_1, consultation_id="pl1")
    path = "minzdrav_protocols/flebo/тромбоз.pdf"
    out = build_consult_alignment(
        doc,
        protocol_paths=[path],
        icd_codes=["I80.1"],
        get_chunks=_fake_chunks,
    )
    exams_crit = next(c for c in out["criteria"] if "обслед" in (c.get("name_ru") or "").lower())
    assert exams_crit.get("item_details") is not None
    assert exams_crit.get("findings_ru")
    assert exams_crit.get("context_ru") is not None or exams_crit.get("name_ru")


def test_completeness_card_sop_comment():
    text = """\
Жалобы: кашель с мокротой три дня, температура до 38.
Анамнез заболевания: болеет 3 дня после переохлаждения.
Анамнез жизни: аллергия на пенициллин.
Объективный статус: АД 120/80, пульс 72, температура 36.6.
Диагноз: J06.8 Острая инфекция верхних дыхательных путей.
"""
    doc = parse_consultation(text, consultation_id="sop2")
    out = build_consult_alignment(
        doc,
        protocol_paths=[],
        icd_codes=["J06.8"],
        get_chunks=lambda _p: [],
    )
    complaints = next(c for c in out["criteria"] if (c.get("name_ru") or "").startswith("Жалобы"))
    assert "Указано" in complaints.get("comment_ru", "") or complaints.get("findings_ru") or "СОП" in complaints.get("comment_ru", "")
    assert complaints.get("protocol_excerpt") or complaints.get("reference_ru")


def test_expand_kz_blob_includes_raw():
    doc = parse_consultation(PL_1, consultation_id="pl2")
    blob = expand_kz_blob(doc, "exams")
    assert "УЗИ" in blob or "ривароксабан" in blob.lower()
