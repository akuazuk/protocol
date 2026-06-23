"""Регрессии по 9 worst cases symptom_icd_probe (r174)."""
from __future__ import annotations

from icd_mkb import suggest_icd_from_russian


def _top(code_list: list[str], n: int = 3) -> list[str]:
    return code_list[:n]


def test_hypertension_not_brain_compression() -> None:
    q = "гипертоническая болезнь давление 170/100"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith(("I10", "I11", "R03"))
    assert "G93.5" not in codes


def test_gastritis_not_sequelae() -> None:
    q = "гастрит изжога после еды"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("K")
    assert "B90" not in codes


def test_dyspepsia_not_sequelae() -> None:
    q = "диспепсия вздутие после еды"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith(("K", "R"))
    assert "B90" not in codes


def test_uti_adult_not_postpartum() -> None:
    q = "инфекция мочевых путей у женщины"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("N")
    assert "O86.2" not in codes
    assert "P39.3" not in codes


def test_alcohol_dependence_not_family_history() -> None:
    q = "алкогольная зависимость абстиненция"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("F10")
    assert "Z81.1" not in codes


def test_menopause_not_empty() -> None:
    q = "климакс приливы потливость"
    codes = [s["code"] for s in suggest_icd_from_russian(q)]
    assert codes
    assert codes[0].startswith(("N95", "E28", "R23"))


def test_urticaria_not_sequelae() -> None:
    q = "крапивница сыпь после еды"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith(("L50", "T78"))
    assert "B90" not in codes


def test_thermal_burn_hand_not_sunburn() -> None:
    q = "ожог руки кипятком"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("T")
    assert codes[0] != "L55"


def test_anaphylaxis_bee_not_rat_fever() -> None:
    q = "анафилаксия после укуса пчелы"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith(("T78", "W57", "L50"))
    assert "A25" not in codes


def test_adenoids_not_infant_death() -> None:
    q = "аденоидит у ребенка храп"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("J35")
    assert codes[0] != "R95"


def test_co_poisoning_not_headache_only() -> None:
    q = "отравление угарным газом головная боль"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith(("T58", "X47", "T59"))


def test_sore_throat_runny_nose_not_y55() -> None:
    q = "болит горло и насморк"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert not any(c.startswith("Y") for c in codes)
    assert codes[0].startswith("J")

    from icd_mkb import analyze_query_for_icd

    analysis = analyze_query_for_icd(q, q)
    retrieval = analysis.get("codes_for_retrieval") or []
    assert not any(str(c).startswith("Y") for c in retrieval)
    assert any(str(c).startswith("J") for c in retrieval)


def test_foreign_body_airway() -> None:
    q = "инородное тело в дыхательных путях"
    codes = _top([s["code"] for s in suggest_icd_from_russian(q)])
    assert codes[0].startswith("T17")
