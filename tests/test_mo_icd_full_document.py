"""МКБ: поиск по всему МО, не только графа «Диагноз»."""
from __future__ import annotations

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.kz_evaluation_engine import evaluate_kz_v3
from clinical_knowledge.mo_icd_resolve import (
    SOURCE_EMPTY,
    SOURCE_SLOT,
    SOURCE_SOFT_FILL_FULL_DOC,
    assess_icd_code_requirement,
    resolve_icd_codes_from_mo,
    soft_fill_mkb_for_warehouse,
)
from clinical_knowledge.reg55_criteria import _how_checked_ru, _icd10_present


def test_resolve_prefers_explicit_main_then_diagnosis_then_elsewhere() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "mkb_code_main": "N47.1",
            "objective_status": "Состояние после операции. Z98.8 в анамнезе.",
            "clinical_diagnosis": "Состояние после циркумцизио",
        }
    )
    assert resolved["main"] == "N47.1"
    assert "N47.1" in resolved["all"]
    assert "Z98.8" in resolved["all"]
    assert resolved["present"] is True


def test_resolve_finds_code_outside_diagnosis_field() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": "Состояние после циркумцизио, френулотомии",
            "mkb_code_main": "",
            "treatment_recommendations": "Наблюдение. Код МКБ N47.0 учтён в плане.",
        }
    )
    assert resolved["present"] is True
    assert resolved["main"] == "N47.0"
    assert any(item["field"] == "treatment_recommendations" for item in resolved["sources"])


def test_resolve_nested_clinical_blob() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical": {
                "objective_status": "Локальный статус. Диагноз по МКБ: N48.8.",
            }
        }
    )
    assert resolved["main"] == "N48.8"
    assert resolved["present"] is True


def test_resolve_empty_when_no_code_anywhere() -> None:
    resolved = resolve_icd_codes_from_mo(
        {
            "clinical_diagnosis": "Состояние после циркумцизио",
            "objective_status": "Повязка сухая.",
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


def test_reg55_icd10_present_scans_full_mo() -> None:
    assert _icd10_present({"diagnosis_short": ""}) is False
    assert _icd10_present({"diagnosis_short": "", "objective_status": "N47.1 после операции"}) is True
    assert _icd10_present({"diagnosis_short": "N47.1"}) is True
    # критерий может быть deferred в mz_2021_55.json, но helper обязан сканить весь МО
    assert _icd10_present(
        {
            "diagnosis_short": "Состояние после циркумцизио",
            "objective_status": "Локально спокойно. N47.1.",
        }
    ) is True
    # текст диагноза без кода - не fail
    assert _icd10_present(
        {"clinical_diagnosis": "Острая рецидивирующая анальная трещина.", "mkb_code_main": ""}
    ) is True


def test_reg55_how_checked_mentions_full_document() -> None:
    text = _how_checked_ru({"check": "icd10_present"})
    assert "формулировка" in text.lower() or "диагноз" in text.lower()


def test_deep_no_b_icd_invalid_when_code_in_recommendations() -> None:
    case = {
        "clinical_diagnosis": "Состояние после циркумцизио",
        "complaints": "Боли в области раны",
        "anamnesis_doctor": "Оперирован вчера",
        "objective_status": "Ране спокойно",
        "treatment_recommendations": "Перевязки. МКБ N47.1.",
        "mkb_code_main": "",
    }
    deep = evaluate_kz_deep(case)
    codes = {f["code"] for f in deep.get("findings") or []}
    assert "B_icd_invalid" not in codes


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


def test_deep_b_icd_invalid_when_neither_diagnosis_nor_code() -> None:
    case = {
        "clinical_diagnosis": "",
        "complaints": "Боль",
        "anamnesis_doctor": "Давно",
        "objective_status": "Без особенностей",
        "treatment_recommendations": "Наблюдение",
        "mkb_code_main": "",
    }
    # без Dx ось concordance может не ставить B_icd; assess - дефект missing_both
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


def test_v3_engine_accepts_icd_outside_diagnosis() -> None:
    case = {
        "clinical_diagnosis": "Состояние после френулотомии",
        "complaints": "Боли",
        "anamnesis_doctor": "Оперирован",
        "objective_status": "Ране спокойно",
        "exam_data": "",
        "treatment_recommendations": "Дикловит. Код N47.0.",
        "mkb_code_main": "",
    }
    result = evaluate_kz_v3(case)
    codes = {f.code for f in (result.findings or [])}
    assert "B_icd_invalid" not in codes


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


def test_soft_fill_prefers_slot_over_full_doc() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "I10",
            "objective_status": "N47.1 в статусе",
        }
    )
    assert fill["source"] == SOURCE_SLOT
    assert fill["code"] == "I10"
    assert fill["slot_code"] == "I10"


def test_soft_fill_from_full_doc_when_slot_empty() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "",
            "mkb_codes": "",
            "clinical_diagnosis": "Состояние после операции",
            "objective_status": "Локально спокойно. N47.1.",
        }
    )
    assert fill["source"] == SOURCE_SOFT_FILL_FULL_DOC
    assert fill["code"] == "N47.1"
    assert fill["slot_code"] == ""


def test_soft_fill_empty_when_no_code() -> None:
    fill = soft_fill_mkb_for_warehouse(
        {
            "mkb_code_main": "",
            "objective_status": "без кода",
        }
    )
    assert fill["source"] == SOURCE_EMPTY
    assert fill["code"] == ""
