"""МКБ: только слоты «Клинический диагноз» / «Диагноз МИС» (не весь МО)."""
from __future__ import annotations

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.kz_evaluation_engine import evaluate_kz_v3
from clinical_knowledge.mo_icd_resolve import (
    SOURCE_EMPTY,
    SOURCE_SLOT,
    SOURCE_SOFT_FILL_DIAG_SLOTS,
    SOURCE_SOFT_FILL_FULL_DOC,
    assess_icd_code_requirement,
    resolve_icd_codes_from_mo,
    soft_fill_mkb_for_warehouse,
)
from clinical_knowledge.reg55_criteria import _how_checked_ru, _icd10_present


def test_resolve_prefers_explicit_main_ignores_non_diag_fields() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "mkb_code_main": "N47.1",
            "objective_status": "Состояние после операции. Z98.8 в анамнезе.",
            "clinical_diagnosis": "Состояние после циркумцизио",
        }
    )
    assert resolved["main"] == "N47.1"
    assert "N47.1" in resolved["all"]
    assert "Z98.8" not in resolved["all"]
    assert resolved["present"] is True


def test_resolve_ignores_code_outside_diagnosis_slots() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": "Состояние после циркумцизио, френулотомии",
            "mkb_code_main": "",
            "treatment_recommendations": "Наблюдение. Код МКБ N47.0 учтён в плане.",
            "objective_status": "Z98.8 после операции",
        }
    )
    assert resolved["present"] is False
    assert resolved["main"] == ""
    assert resolved["all"] == []


def test_resolve_from_mis_diagnos_slot() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": (
                "Внебольничная нижнедолевая полисегментарная пневмония, "
                "средней тяжести. ДН0"
            ),
            "mis_diagnos": "J18.9",
            "mkb_code_main": "",
            "anamnesis_doctor": "Ранее Z98.8, гипертония I10.",
        }
    )
    assert resolved["main"] == "J18.9"
    assert resolved["all"] == ["J18.9"]
    assert any(item["field"] == "mis_diagnos" for item in resolved["sources"])


def test_resolve_nested_clinical_diagnosis_only() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical": {
                "clinical_diagnosis": "Локальный статус. Диагноз по МКБ: N48.8.",
                "objective_status": "Z98.8 в статусе не брать",
            }
        }
    )
    assert resolved["main"] == "N48.8"
    assert "Z98.8" not in resolved["all"]
    assert resolved["present"] is True


def test_resolve_empty_when_no_code_in_diag_slots() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": "Состояние после циркумцизио",
            "objective_status": "Повязка сухая. N47.1.",
        }
    )
    assert resolved["present"] is False
    assert resolved["main"] == ""
    assert resolved["all"] == []


def test_resolve_cyrillic_h_with_space_myopia_visit() -> None:
    """Визит 3644142: «Н 52.1» / «Н 52.2» - кириллица + пробел, не «нет кода»."""
    text = (
        "Н 52.1 Миопия слабой степени обоих глаз.\r\n"
        "Н 52.2 Миопический астигматизм обоих глаз."
    )
    resolved = resolve_icd_codes_from_mo({"clinical_diagnosis": text, "mkb_code_main": ""})
    assert resolved["present"] is True
    assert resolved["main"] == "H52.1"
    assert "H52.1" in resolved["all"]
    assert "H52.2" in resolved["all"]


def test_deep_no_b_icd_invalid_on_cyrillic_h_codes() -> None:
    case = {
        "clinical_diagnosis": (
            "Н 52.1 Миопия слабой степени обоих глаз.\r\n"
            "Н 52.2 Миопический астигматизм обоих глаз."
        ),
        "mkb_code_main": "",
        "complaints": "Снижение зрения",
        "anamnesis_doctor": "Длительно",
        "objective_status": "OU миопия",
        "exam_recommendations": "Контроль",
        "treatment_recommendations": "Коррекция",
    }
    deep = evaluate_kz_deep(case, protocol_ctx=None, drug_ctx={})
    codes = {f["code"] for f in (deep.get("findings") or [])}
    assert "B_icd_invalid" not in codes


def test_reg55_icd10_present_uses_diag_slots_only() -> None:
    assert _icd10_present({"diagnosis_short": ""}) is False
    # код только в objective - больше не считается
    assert _icd10_present({"diagnosis_short": "", "objective_status": "N47.1 после операции"}) is False
    assert _icd10_present({"diagnosis_short": "N47.1"}) is True
    assert _icd10_present({"mis_diagnos": "J18.9", "clinical_diagnosis": "Пневмония"}) is True
    # текст диагноза без кода - не fail
    assert _icd10_present(
        {"clinical_diagnosis": "Острая рецидивирующая анальная трещина.", "mkb_code_main": ""}
    ) is True


def test_reg55_how_checked_mentions_diag_slots() -> None:
    text = _how_checked_ru({"check": "icd10_present"})
    assert "клинический диагноз" in text.lower()
    assert "диагноз мис" in text.lower()


def test_deep_no_b_icd_invalid_when_diagnosis_without_code() -> None:
    case = {
        "clinical_diagnosis": "Острая рецидивирующая анальная трещина.",
        "complaints": "Боль",
        "anamnesis_doctor": "Рецидив",
        "objective_status": "Трещина",
        "treatment_recommendations": "Диета",
        "mkb_code_main": "",
    }
    deep = evaluate_kz_deep(case)
    codes = {f["code"] for f in deep.get("findings") or []}
    assert "B_icd_invalid" not in codes
    assess = assess_icd_code_requirement(case)
    assert assess["ok"] is True
    assert assess["status"] == "diagnosis_without_code"


def test_code_only_in_plan_does_not_satisfy_icd_present_helper() -> None:
    case = {
        "clinical_diagnosis": "Состояние после циркумцизио",
        "treatment_recommendations": "Перевязки. МКБ N47.1.",
        "mkb_code_main": "",
    }
    assert resolve_icd_codes_from_mo(case)["present"] is False
    # есть текст Dx - assess ok; код из плана не подхватывается
    assess = assess_icd_code_requirement(case)
    assert assess["ok"] is True
    assert assess["status"] == "diagnosis_without_code"


def test_deep_b_icd_invalid_when_neither_diagnosis_nor_code() -> None:
    case = {
        "clinical_diagnosis": "",
        "complaints": "Боль",
        "anamnesis_doctor": "Давно",
        "objective_status": "Без особенностей",
        "treatment_recommendations": "Наблюдение",
        "mkb_code_main": "",
    }
    _ = evaluate_kz_deep(case)
    assess = assess_icd_code_requirement(case)
    assert assess["ok"] is False
    assert assess["status"] == "missing_both"


def test_b_icd_invalid_on_malformed_explicit_code() -> None:
    case = {
        "clinical_diagnosis": "Острая рецидивирующая анальная трещина.",
        "complaints": "Боль",
        "anamnesis_doctor": "Рецидив",
        "objective_status": "Трещина",
        "mkb_code_main": "XXX",
    }
    assess = assess_icd_code_requirement(case)
    assert assess["ok"] is False
    assert assess["status"] == "invalid_format"
    deep = evaluate_kz_deep(case)
    codes = {f["code"] for f in deep.get("findings") or []}
    assert "B_icd_invalid" in codes


def test_v3_engine_no_b_icd_invalid_when_diagnosis_without_code() -> None:
    case = {
        "clinical_diagnosis": "Острая рецидивирующая анальная трещина.",
        "complaints": "Боль при дефекации",
        "anamnesis_doctor": "Рецидивирует",
        "objective_status": "Трещина анального канала",
        "exam_data": "",
        "treatment_recommendations": "Диета, свечи",
        "mkb_code_main": "",
    }
    result = evaluate_kz_v3(case)
    codes = {f.code for f in (result.findings or [])}
    assert "B_icd_invalid" not in codes


def test_soft_fill_prefers_slot_over_diag_text() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "I10",
            "objective_status": "N47.1 в статусе",
            "clinical_diagnosis": "N47.1 что-то",
        }
    )
    assert fill["source"] == SOURCE_SLOT
    assert fill["code"] == "I10"
    assert fill["slot_code"] == "I10"


def test_soft_fill_from_diag_slots_when_explicit_empty() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "",
            "mkb_codes": "",
            "clinical_diagnosis": "Состояние после операции N47.1",
            "objective_status": "Локально спокойно. Z98.8.",
        }
    )
    assert fill["source"] in {SOURCE_SOFT_FILL_DIAG_SLOTS, SOURCE_SOFT_FILL_FULL_DOC}
    assert fill["code"] == "N47.1"
    assert fill["slot_code"] == ""


def test_soft_fill_empty_when_code_only_outside_diag() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "",
            "clinical_diagnosis": "Состояние после операции",
            "objective_status": "N47.1",
        }
    )
    assert fill["source"] == SOURCE_EMPTY
    assert fill["code"] == ""
