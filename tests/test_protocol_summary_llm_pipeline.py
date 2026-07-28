"""Tests for protocol summary LLM pipeline helpers."""
from __future__ import annotations

import pytest

from clinical_knowledge.protocol_summary.llm_json import parse_json_loose
from clinical_knowledge.protocol_summary.llm_extractor import extract_protocol_summary_llm
from clinical_knowledge.protocol_summary.llm_merger import build_condition_summary, merge_to_protocol_summary
from clinical_knowledge.protocol_summary.quote_validator import quote_found_in_source, validate_quotes_in_payload
from clinical_knowledge.protocol_summary.source_text import build_source_text_document, section_text_blob
from clinical_knowledge.protocol_summary.summary_to_rules import summary_to_protocol_rules


def test_parse_json_loose_strips_fence():
    raw = '```json\n{"a": 1}\n```'
    assert parse_json_loose(raw) == {"a": 1}


def test_quote_found_in_source_fuzzy():
    blob = "Пациенту назначается эзофагогастродуоденоскопия при подозрении на эрозию"
    assert quote_found_in_source("эзофагогастродуоденоскопия при подозрении", blob)


def test_validate_quotes_flags_missing():
    issues = validate_quotes_in_payload(
        {"required_exams": [{"name": "КТ", "quote": "абсолютно выдуманная цитата xyz"}]},
        "только реальный текст протокола",
    )
    assert issues


def test_merge_to_protocol_summary_minimal():
    doc = {
        "protocol_id": "test_proto",
        "path": "corpus/test.pdf",
        "title": "Тестовый протокол",
        "specialty_slug": "test",
        "audience": "adult",
    }
    skeleton = {
        "title_ru": "Тестовый протокол",
        "population": ["adult"],
        "conditions": [{"condition_id": "gastritis", "name": "Гастрит", "icd10_codes": ["K29.7"]}],
    }
    blocks = {
        "gastritis": [
            {
                "required_exams": [{"name": "ЭГДС", "level": "required", "quote": "ЭГДС обязательна", "page_start": 3}],
                "diagnostic_criteria": ["боль в эпигастрии"],
                "drugs": [],
                "red_flags": [],
            }
        ]
    }
    summary = merge_to_protocol_summary(doc, skeleton, blocks, extractor="test", extractor_version="0")
    assert summary.protocol_id == "test_proto"
    assert summary.conditions[0].icd10_codes == ["K29.7"]
    assert summary.conditions[0].required_exams[0].name == "ЭГДС"


def test_build_condition_summary_from_block():
    sk = {"condition_id": "c1", "name": "А", "icd10_codes": ["J06.9"]}
    block = {"required_exams": [{"name": "ОАК", "level": "required", "quote": "ОАК"}]}
    cond = build_condition_summary("pid", "corpus/x.pdf", sk, [block])
    assert cond.condition_id == "c1"
    assert cond.required_exams[0].name == "ОАК"


def test_build_condition_summary_keeps_full_drug_regimen():
    sk = {"condition_id": "c1", "name": "А", "icd10_codes": ["J06.9"]}
    block = {
        "drugs": [{
            "name": "Амоксициллин",
            "dose_text": "500 мг",
            "route": "перорально",
            "frequency_text": "3 раза в сутки",
            "duration_text": "7 дней",
            "monitoring": ["оценить эффект через 48-72 часа"],
            "quote": "Амоксициллин 500 мг перорально 3 раза в сутки 7 дней",
            "page_start": 4,
        }],
    }
    cond = build_condition_summary("pid", "corpus/x.pdf", sk, [block])
    drug = cond.treatment.drugs[0]
    assert drug.drug_name == "Амоксициллин"
    assert drug.dose_text == "500 мг"
    assert drug.route == "перорально"
    assert drug.frequency_text == "3 раза в сутки"
    assert drug.duration_text == "7 дней"
    assert drug.monitoring == ["оценить эффект через 48-72 часа"]


def test_strict_extraction_rejects_empty_model_result(monkeypatch):
    monkeypatch.setenv("PROTOCOL_LLM_STRICT", "1")
    doc = {
        "protocol_id": "strict_proto",
        "path": "corpus/strict.pdf",
        "title": "Тест",
        "sections": {},
    }
    with pytest.raises(RuntimeError, match="Не получена структура"):
        extract_protocol_summary_llm(
            doc,
            use_llm=True,
            model=object(),
            generate_fn=lambda _model, _prompt: "{}",
        )


def test_pregnant_population_becomes_required_applicability():
    doc = {
        "protocol_id": "pregnancy_proto",
        "path": "corpus/pregnancy.pdf",
        "title": "Ведение беременности",
        "specialty_slug": "akusherstvo-ginekologiya",
    }
    skeleton = {
        "title_ru": "Ведение беременности",
        "population": ["pregnant"],
        "conditions": [{
            "condition_id": "pregnancy",
            "name": "Беременность",
            "icd10_codes": ["O26.9"],
        }],
    }
    blocks = {
        "pregnancy": [{
            "required_exams": [{
                "name": "Общий анализ крови",
                "level": "required",
                "quote": "Общий анализ крови",
            }],
        }],
    }
    summary = merge_to_protocol_summary(
        doc,
        skeleton,
        blocks,
        extractor="test",
        extractor_version="0",
    )
    rule = summary_to_protocol_rules(summary)[0]
    assert rule.applicability.pregnancy == "required"
    assert rule.applicability.sex == "female"
    assert "adult" in rule.applicability.age_groups


def test_section_text_blob_from_doc():
    doc = {
        "sections": {
            "diagnostics": [{"section_title": "Диагностика", "text": "Назначается УЗИ органов брюшной полости"}],
        }
    }
    blob = section_text_blob(doc, ["diagnostics"])
    assert "УЗИ" in blob
