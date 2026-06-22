"""Тесты детерминированных карточек согласования КЗ."""
from __future__ import annotations

from clinical_knowledge.consult_alignment import (
    _mkb_reference_line,
    build_consult_alignment,
    merge_alignment_into_review,
)
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.dispensary_regulations import lookup_follow_up_expectations

MG_1 = """\
Врач: терапевт
Дата консультации: 01.02.2024
Пол: мужской
Жалобы: кашель, насморк.
Анамнез: болеет 3 дня.
Объективный статус: удовлетворительное.
Диагноз: J06.8 Другие острые инфекции верхних дыхательных путей.
Рекомендации по лечению: симптоматическая терапия.
Контроль через неделю.
"""

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
                "chunk_type": "diagnostics",
                "text": "УЗИ вен нижних конечностей - обязательное исследование при флеботромбозе.",
                "section_title": "Диагностика",
                "page_from": 12,
                "icd10_codes": ["I80.1"],
            },
            {
                "chunk_type": "pharmacotherapy",
                "text": "Ривароксабан 20 мг 1 раз в сутки - прямые оральные антикоагулянты.",
                "section_title": "Лечение",
                "page_from": 24,
                "icd10_codes": ["I80.1"],
            },
            {
                "chunk_type": "dispensary",
                "text": "Контрольное УЗИ вен через 3 месяца, повторная консультация флеболога.",
                "section_title": "Наблюдение",
                "page_from": 30,
            },
        ]
    return []


def test_mkb_reference_line_no_duplicate_code():
    assert _mkb_reference_line("N72", "N72 - Воспалительная болезнь шейки матки") == (
        "N72 - Воспалительная болезнь шейки матки"
    )
    assert _mkb_reference_line("E03.9", "Гипотиреоз неуточненный") == "E03.9 - Гипотиреоз неуточненный"


def test_diagnosis_card_uses_mkb_not_kp():
    doc = parse_consultation(MG_1, consultation_id="mg1")
    out = build_consult_alignment(
        doc,
        protocol_paths=[],
        icd_codes=["J06.8"],
        get_chunks=lambda _p: [],
    )
    diag = next(c for c in out["alignment_cards"] if c["block_id"] == "diagnosis")
    assert diag["source_kind"] == "mkb"
    assert diag["name_ru"] == "Диагноз и коды (МКБ-10)"
    assert diag["score_pct"] >= 70
    assert "МКБ" in diag["protocol_section"] or diag["protocol_excerpt"]


def test_complaints_not_compared_to_kp():
    doc = parse_consultation(MG_1, consultation_id="mg1")
    out = build_consult_alignment(doc, protocol_paths=[], icd_codes=["J06.8"], get_chunks=lambda _p: [])
    complaints = next(c for c in out["alignment_cards"] if c["block_id"] == "complaints")
    assert complaints["source_kind"] == "completeness"
    assert "жалоб" in complaints["comment_ru"].lower()
    assert "отдельн" not in complaints["comment_ru"].lower()
    assert complaints.get("reference_ru") or complaints.get("protocol_excerpt")
    assert "СОП" in (complaints.get("protocol_excerpt") or complaints.get("reference_ru") or "")


def test_anamnesis_separate_from_complaints():
    text = """\
Жалобы: кашель 3 дня.
Анамнез заболевания: ОРВИ переносил в детстве, текущее 3 дня.
Анамнез жизни: аллергия на пенициллин.
Диагноз: J06.8
"""
    doc = parse_consultation(text, consultation_id="an1")
    out = build_consult_alignment(doc, protocol_paths=[], icd_codes=["J06.8"], get_chunks=lambda _p: [])
    anam = next(c for c in out["alignment_cards"] if c["block_id"] == "anamnesis")
    assert "Заболевание:" in (anam.get("conclusion_excerpt") or "")
    assert "Жизни:" in (anam.get("conclusion_excerpt") or "")
    assert "отделён" not in anam["comment_ru"].lower()
    assert "анамнез заболевания" in anam["comment_ru"].lower()


def test_exams_context_findings_gaps():
    doc = parse_consultation(PL_1, consultation_id="pl1")
    path = "minzdrav_protocols/flebo/тромбоз.pdf"
    matches = [{"title": "Флеботромбоз", "source_path": path, "match_score": 78.0}]
    out = build_consult_alignment(
        doc,
        protocol_paths=[path],
        icd_codes=["I80.1"],
        get_chunks=_fake_chunks,
        protocol_matches=matches,
        specialty_label="флеболог",
    )
    exams = next(c for c in out["alignment_cards"] if c["block_id"] == "exams")
    assert exams["source_kind"] == "kp"
    assert "Флеботромбоз" in exams["comment_ru"] or "КП" in exams["comment_ru"]
    crit = next(c for c in out["criteria"] if "обслед" in (c.get("name_ru") or "").lower())
    assert crit.get("context_ru")
    assert crit.get("findings_ru") or crit.get("gaps_ru") is not None


def test_exams_treatment_from_kp_chunks():
    doc = parse_consultation(PL_1, consultation_id="pl1")
    path = "minzdrav_protocols/flebo/тромбоз.pdf"
    out = build_consult_alignment(
        doc,
        protocol_paths=[path],
        icd_codes=["I80.1"],
        get_chunks=_fake_chunks,
    )
    exams = next(c for c in out["alignment_cards"] if c["block_id"] == "exams")
    treat = next(c for c in out["alignment_cards"] if c["block_id"] == "treatment")
    assert exams["source_kind"] == "kp"
    assert treat["source_kind"] == "kp"
    assert exams["score_pct"] >= 50


def test_follow_up_regulation_lookup():
    reg = lookup_follow_up_expectations(["I80.1"])
    assert reg.get("regulation_id") == "mz_2015_127"
    assert reg.get("follow_up_hints")


def test_merge_alignment_into_review():
    review = {"criteria": [{"name_ru": "LLM", "score_pct": 50}], "summary_ru": "ok"}
    alignment = {
        "criteria": [{"name_ru": "Диагноз", "score_pct": 90, "source_kind": "mkb"}],
        "alignment_cards": [],
        "limitations_ru": "test",
    }
    merge_alignment_into_review(review, alignment)
    assert review["criteria_source"] == "deterministic_alignment"
    assert review["criteria"][0]["name_ru"] == "Диагноз"
