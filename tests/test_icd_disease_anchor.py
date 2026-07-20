"""Доводка ICD-лексикона: название болезни важнее симптом-кодов (R/J06/онко)."""
from __future__ import annotations

import pytest

from icd_mkb import suggest_icd_from_russian


def _top_codes(query: str, n: int = 3) -> list[str]:
    return [s["code"] for s in suggest_icd_from_russian(query, max_results=8)][:n]


@pytest.mark.parametrize(
    "query,expected_prefixes,forbidden",
    [
        ("внебольничная пневмония кашель с мокротой лихорадка", ("J18", "J15", "J13", "J14"), ("R50", "J06.9")),
        ("хроническая обструктивная болезнь легких одышка кашель курильщик", ("J44", "J43"), ("R05", "J06.9")),
        ("острый тонзиллит боль в горле налеты на миндалинах", ("J03", "J02", "J35"), ("R07.0",)),
        ("острый пиелонефрит боль в пояснице температура", ("N10", "N11", "N12"), ("R50", "J06.9")),
        ("язвенная болезнь желудка боль натощак", ("K25", "K26", "K27"), ("C16", "D00", "D13")),
        ("мигрень сильная головная боль светобоязнь", ("G43", "G44"), ("R51",)),
        ("атопический дерматит зуд и сыпь у ребенка", ("L20", "L23", "L30"), ("L50", "T78")),
    ],
)
def test_disease_anchor_top1(query, expected_prefixes, forbidden) -> None:
    codes = _top_codes(query, 3)
    assert codes, "ожидались коды"
    assert any(codes[0].upper().startswith(p) for p in expected_prefixes), (
        f"top-1 {codes[0]} не из {expected_prefixes}; список {codes}"
    )
    # опасный/симптомный код не должен быть top-1
    assert not any(codes[0].upper().startswith(f) for f in forbidden), (
        f"top-1 {codes[0]} - шум {forbidden}"
    )


def test_peptic_ulcer_not_neoplasm_in_top3() -> None:
    codes = _top_codes("язвенная болезнь желудка боль натощак", 3)
    for c in codes:
        assert not c.upper().startswith(("C16", "D00")), f"онкокод {c} в top-3 при язве"
