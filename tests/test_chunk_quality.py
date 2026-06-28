"""Тесты chunk_quality и chunk_type_infer."""
from clinical_knowledge.chunk_quality import (
    ISSUE_ICD_INFLATION,
    ISSUE_PREAMBLE_LEAK,
    ISSUE_TYPE_BODY_BUT_CLINICAL,
    detect_issues,
    quality_score,
    should_index,
    strip_noise_lines,
)
from clinical_knowledge.chunk_type_infer import infer_chunk_type, resolve_section_title


def test_preamble_not_indexable() -> None:
    ch = {
        "chunk_type": "body",
        "text": "Постановление Министерства здравоохранения об утверждении клинического протокола",
        "tags": {"signal": "low", "is_preamble": True},
        "indexable": False,
    }
    assert should_index(ch) is False
    assert ISSUE_PREAMBLE_LEAK in detect_issues(ch)


def test_clinical_body_detected() -> None:
    ch = {
        "chunk_type": "body",
        "text": "Рекомендуется выполнить общий анализ крови и УЗИ органов брюшной полости при обострении.",
        "tags": {"signal": "medium"},
    }
    assert ISSUE_TYPE_BODY_BUT_CLINICAL in detect_issues(ch)
    suggested = infer_chunk_type(section_title="1. Диагностика", text=ch["text"])
    assert suggested == "diagnostics"


def test_icd_inflation_detected() -> None:
    ch = {
        "chunk_type": "diagnostics",
        "text": "лабораторные исследования",
        "icd10_codes": [f"A{i:02d}" for i in range(20)],
    }
    assert ISSUE_ICD_INFLATION in detect_issues(ch)


def test_strip_noise() -> None:
    raw = "Утверждено\nРекомендуется назначить ОАК\nСогласовано"
    cleaned = strip_noise_lines(raw)
    assert "Утверждено" not in cleaned
    assert "ОАК" in cleaned


def test_resolve_weak_section_title() -> None:
    title = resolve_section_title("таблица", ["1. Диагностика", "таблица"])
    assert title == "1. Диагностика"


def test_quality_score_range() -> None:
    good = {
        "chunk_type": "diagnostics",
        "section_title": "1. Диагностика",
        "text": "Рекомендуется выполнить ОАК, БАК, УЗИ при постановке диагноза и каждые 3 месяца.",
        "icd10_codes": ["D68.6"],
        "lab_tests": ["ОАК"],
        "tags": {"signal": "high"},
        "indexable": True,
    }
    assert quality_score(good) >= 0.85
