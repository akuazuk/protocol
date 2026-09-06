"""Обезличивание перед Gemini: идентификаторы не должны попадать в промпт.

Gemini - обработчик за пределами Беларуси, поэтому в запросе остаётся только
то, без чего задача не решается. Клинический текст остаётся (он и есть предмет
оценки), прямые идентификаторы - нет.
"""
from __future__ import annotations

from datetime import date

import pytest

from clinical_knowledge.phi_for_llm import (
    age_from_dob,
    contains_identifier_label,
    pseudonym,
    redact_mapping_for_llm,
    redact_text_for_llm,
)


def test_pseudonym_is_stable_and_not_reversible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один случай - один псевдоним, но исходный номер в нём не виден.

    Устойчивость нужна методисту: два разбора одного случая должны совпадать.
    """
    monkeypatch.setenv("PHI_PSEUDONYM_KEY", "тестовый-ключ")
    first = pseudonym("3646270", prefix="case")
    assert first == pseudonym("3646270", prefix="case")
    assert first != pseudonym("3646271", prefix="case")
    assert "3646270" not in first
    assert first.startswith("case-")


def test_pseudonym_depends_on_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHI_PSEUDONYM_KEY", "ключ-один")
    with_first = pseudonym("3646270")
    monkeypatch.setenv("PHI_PSEUDONYM_KEY", "ключ-два")
    assert pseudonym("3646270") != with_first


def test_pseudonym_keeps_empty_empty() -> None:
    assert pseudonym("") == ""
    assert pseudonym(None) == ""


def test_age_from_dob_formats() -> None:
    on = date(2026, 9, 5)
    assert age_from_dob("12.07.1976", on=on) == 50
    assert age_from_dob("1976-07-12", on=on) == 50
    assert age_from_dob(date(2020, 12, 31), on=on) == 5
    assert age_from_dob("не дата", on=on) is None


def test_redact_text_hides_dob_but_keeps_age() -> None:
    """Для выбора протокола нужен возраст; точная дата - лишний идентификатор."""
    text = "Дата рождения: 12.07.1976\nЖалобы: боль в горле, температура 38."
    out = redact_text_for_llm(text, on=date(2026, 9, 5))

    assert "12.07.1976" not in out
    assert "возраст 50 лет" in out
    # Клиника обязана сохраниться полностью.
    assert "боль в горле" in out
    assert "температура 38" in out


def test_redact_text_hides_phone() -> None:
    out = redact_text_for_llm("Телефон: +375 29 123 45 67\nЖалобы: кашель")
    assert "123 45 67" not in out
    assert "[телефон скрыт]" in out
    assert "кашель" in out


def test_redact_text_reduces_names_to_initials() -> None:
    out = redact_text_for_llm("Пациент: Иванов Пётр Сергеевич\nЖалобы: кашель")
    assert "Иванов" not in out
    assert "кашель" in out


def test_redact_mapping_replaces_identifiers_and_names() -> None:
    row = {
        "visit_id": "3646270",
        "patient_id": "88123",
        "mis_id": "M-1",
        "doctor_fio": "Иванов Пётр Сергеевич",
        # Клинические и контекстные поля не трогаем.
        "age_years": 50,
        "queue_reason": "нет плана наблюдения",
        "overall_pct_system": 62,
    }
    out = redact_mapping_for_llm(row)

    for raw in ("3646270", "88123", "M-1", "Иванов"):
        assert raw not in str(out), f"{raw} утёк в промпт"
    assert out["age_years"] == 50
    assert out["queue_reason"] == "нет плана наблюдения"
    assert out["overall_pct_system"] == 62
    assert out["doctor_fio"] == "И. П. С."


def test_full_methodist_prompt_has_no_identifiers() -> None:
    """Самый тяжёлый промпт: раньше нёс ФИО врача, Visit ID и Patient ID."""
    from clinical_knowledge.mis_kz_quality import _build_full_llm_prompt

    row = {
        "doctor_fio": "Иванов Пётр Сергеевич",
        "doctor_specialization": "терапевт",
        "filial": "Филиал 1",
        "date": "2026-07-14 10:00:00",
        "patient_id": "88123",
        "visit_id": "3646270",
    }
    prompt = _build_full_llm_prompt(
        row=row,
        visit_id="3646270",
        text="Дата рождения: 12.07.1976\nЖалобы: боль в горле",
        l2_ctx={},
    )

    assert not contains_identifier_label(prompt), contains_identifier_label(prompt)
    for raw in ("3646270", "88123", "Иванов", "12.07.1976"):
        assert raw not in prompt, f"{raw} остался в промпте методиста"
    # Клинически значимое сохраняется.
    assert "терапевт" in prompt
    assert "боль в горле" in prompt


def test_action_judge_prompts_have_no_identifiers() -> None:
    from clinical_knowledge.mo_llm_action_judge import build_prompt_a, build_prompt_b

    pack = {
        "meta": {
            "case_id": "3646270",
            "visit_id": "3646270",
            "mis_id": "M-1",
            "age_years": 50,
            "queue_reason": "нет плана наблюдения",
        },
        "slots": {"complaints": "боль в горле", "clinical_diagnosis": "J02.9"},
        "plan_slots": {"treatment_recommendations": "полоскание"},
    }

    for prompt in (build_prompt_a(pack), build_prompt_b(pack, {})):
        for raw in ("3646270", "M-1"):
            assert raw not in prompt, f"{raw} остался в промпте судьи"
        assert "боль в горле" in prompt


def test_stage_validation_trusts_caller_not_model() -> None:
    """Эхо идентификатора из ответа модели не должно определять, к какому случаю подшить разбор."""
    from clinical_knowledge.mo_llm_action_judge import EXAMPLE_STAGE_A, validate_stage_a

    # Модель вернула псевдоним (он и был в промпте) - подшить обязаны к настоящему случаю.
    raw = dict(EXAMPLE_STAGE_A)
    raw["case_id"] = "case-deadbeef"
    raw["visit_id"] = "case-deadbeef"

    out = validate_stage_a(raw, case_id="3646270")
    assert out["case_id"] == "3646270"
    assert out["visit_id"] == "3646270"
    assert "deadbeef" not in str(out)


def test_case_narrative_prompt_has_no_identifiers() -> None:
    from clinical_knowledge.mo_case_narrative import build_narrative_prompt

    prompt = build_narrative_prompt(
        {"visit_id": "3646270", "patient_id": "88123", "doctor": "Иванов Пётр Сергеевич"},
        {"complaints": "боль в горле"},
    )
    for raw in ("3646270", "88123", "Иванов"):
        assert raw not in prompt, f"{raw} остался в нарративе"
    assert "боль в горле" in prompt


def test_kz_grader_prompt_redacts_header() -> None:
    from scripts.grade_kz_llm import build_grader_prompt

    case = {
        "complaints": "Пациент: Иванов Пётр Сергеевич\nболь в горле",
        "clinical_diagnosis": "J02.9",
    }
    prompt = build_grader_prompt(case, [("A_complaints", "жалобы конкретны")])
    assert "Иванов" not in prompt
    assert "боль в горле" in prompt
