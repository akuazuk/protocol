"""Тесты разделения жалоб/анамнеза и фильтров подбора КП."""
from __future__ import annotations

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.kz_clinical_context import build_clinical_context, split_anamnesis_parts
from clinical_knowledge.protocol_pick_filters import (
    clinical_relevance_multiplier,
    is_administrative_protocol,
)


NEURO_SAMPLE = """\
Врач: невролог
Дата консультации: 15.06.2026
Пол: мужской, 26 лет
Жалобы: на онемение в бедре слева Aнамнез: Аллергоанамнез не отягощен Онкоанамнез не отягощен Оперативные вмешательства отрицает
Объективный статус: Вес 82 кг. Рост 179 см.
Диагноз: M54.3 Ишиас; Вертеброгенная люмбоишалгия слева.
Рекомендации по лечению: кап. Терафлекс ультра.
"""


def test_inline_anamnesis_split_from_complaints():
    doc = parse_consultation(NEURO_SAMPLE, consultation_id="n1")
    assert "онемение" in (doc.sections.complaints or "").lower()
    assert "аллерго" not in (doc.sections.complaints or "").lower()
    assert "аллерго" in (doc.sections.life_history or "").lower()
    ctx = build_clinical_context(doc, ["M54.3"], specialty_label="невролог")
    assert "Aнамнез" not in (ctx.get("complaints") or "")
    anam = split_anamnesis_parts(doc)
    assert anam["life"]


def test_admin_protocol_filtered():
    card = {
        "title": "Об утверждении клинических протоколов",
        "source_path": "minzdrav/x.pdf",
        "icd10_primary": [],
    }
    assert is_administrative_protocol(card)


def test_m54_penalizes_epilepsy_protocol():
    epilepsy = {
        "title": "Диагностика и лечение пациентов с эпилепсией",
        "source_path": "minzdrav/nevrologiya/epilepsy.pdf",
        "icd10_primary": ["G40"],
    }
    spine = {
        "title": "Диагностика и лечение ишиаса и радикулопатии",
        "source_path": "minzdrav/nevrologiya/ishiас.pdf",
        "icd10_primary": ["M54.3"],
    }
    m_bad = clinical_relevance_multiplier(epilepsy, icd_codes=["M54.3"], ambulatory=True)
    m_good = clinical_relevance_multiplier(spine, icd_codes=["M54.3"], ambulatory=True)
    assert m_good > m_bad
