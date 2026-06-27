"""Сверка финмодели конкурса: формулы, монотонность чувствительности, согласованность таблиц."""
from __future__ import annotations

from konkurs_finance import (
    B2C_AVG_PRICE,
    FIN_Y1,
    FIN_Y2,
    FIN_Y3,
    KRAVIRA_B2B_YEAR,
    KRAVIRA_KZ_MONTH,
    KRAVIRA_MARKET_SHARE,
    MARKET_KZ_MONTH,
    MARKET_KZ_YEAR,
    PRICE_BLEND_Y2Y3,
    TAM_B2B_CEILING_YEAR_K,
    ebitda_k,
    ebitda_month_k,
    total_rev_k,
)
from konkurs_scenarios import (
    B2C_CONV_SENSITIVITY,
    B2C_PROTOCOL_PER_CHECK,
    SCENARIO_CAUTIOUS,
    b2b_k_from_kz_month,
    b2c_protocol_byn_per_check,
    b2c_protocol_k,
    ebitda_sensitivity_b2b_share,
)


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _status(ok: bool) -> str:
    return "OK" if ok else "ОШИБКА"


def run_financial_audit() -> list[tuple[str, str, str, str]]:
    """Полная сверка ключевых расчётов для глоссария и CI."""
    rows: list[tuple[str, str, str, str]] = []

    # --- TAM / якорь ---
    tam_ok = MARKET_KZ_MONTH == int(KRAVIRA_KZ_MONTH / KRAVIRA_MARKET_SHARE)
    rows.append(("TAM КЗ/мес", _fmt_int(MARKET_KZ_MONTH), "25 000 / 1%", _status(tam_ok)))
    rows.append(("TAM КЗ/год", _fmt_int(MARKET_KZ_YEAR), "TAM мес × 12", _status(MARKET_KZ_YEAR == MARKET_KZ_MONTH * 12)))
    rows.append(
        (
            "TAM B2B потолок, тыс.",
            _fmt_int(TAM_B2B_CEILING_YEAR_K),
            f"{_fmt_int(MARKET_KZ_YEAR)} × {PRICE_BLEND_Y2Y3} / 1000",
            _status(TAM_B2B_CEILING_YEAR_K == int(MARKET_KZ_YEAR * PRICE_BLEND_Y2Y3 / 1000)),
        )
    )
    rows.append(
        (
            "Кравира B2B, тыс.",
            str(KRAVIRA_B2B_YEAR // 1000),
            "25k×0,69×12/1000",
            _status(KRAVIRA_B2B_YEAR == 207_000),
        )
    )

    # --- B2C unit economics ---
    calc_per_check = b2c_protocol_byn_per_check()
    rows.append(
        (
            "B2C чек пациента",
            f"{B2C_AVG_PRICE} BYN",
            "микс tier",
            _status(abs(B2C_AVG_PRICE - 8.33) < 0.01),
        )
    )
    rows.append(
        (
            "B2C Protocol/проверка",
            f"{B2C_PROTOCOL_PER_CHECK} BYN",
            f"80%×{B2C_AVG_PRICE}×70% + 20%×{B2C_AVG_PRICE} = {calc_per_check:.3f}",
            _status(abs(B2C_PROTOCOL_PER_CHECK - calc_per_check) < 0.001),
        )
    )

    # --- P&L по годам ---
    for label, fin in [("2027", FIN_Y1), ("2028", FIN_Y2), ("2029", FIN_Y3)]:
        calc_b2c = b2c_protocol_k(fin["b2c_checks"])
        calc_rev = fin["b2b_k"] + calc_b2c + fin.get("api_k", 0)
        calc_ebitda = calc_rev - fin["opex_k"]
        ok = calc_b2c == fin["b2c_k"] and calc_rev == total_rev_k(fin) and calc_ebitda == ebitda_k(fin)
        rows.append(
            (
                f"Выручка {label}, тыс.",
                str(total_rev_k(fin)),
                f"B2B {fin['b2b_k']} + B2C {fin['b2c_k']} + API {fin.get('api_k', 0)}",
                _status(ok),
            )
        )
        rows.append(
            (
                f"EBITDA {label}, тыс./мес",
                f"{ebitda_k(fin)} / {ebitda_month_k(fin)}",
                f"выручка − OPEX {fin['opex_k']}",
                _status(ok),
            )
        )

    # --- B2C sensitivity: монотонность и формула ---
    prev_checks = -1
    prev_ebitda = -1
    mono_ok = True
    formula_ok = True
    b2b_fixed = b2b_k_from_kz_month(int(MARKET_KZ_MONTH * 0.08))
    api_fixed = SCENARIO_CAUTIOUS["api_k"]
    opex_fixed = SCENARIO_CAUTIOUS["opex_k"]
    for conv_s, checks_s, b2c_s, rev_s, ebitda_s in B2C_CONV_SENSITIVITY:
        checks = int(checks_s.replace(" ", ""))
        b2c = int(b2c_s)
        rev = int(rev_s)
        ebitda = int(ebitda_s)
        if checks <= prev_checks or ebitda < prev_ebitda:
            mono_ok = False
        prev_checks = checks
        prev_ebitda = ebitda
        calc_b2c = b2c_protocol_k(checks)
        calc_rev = b2b_fixed + calc_b2c + api_fixed
        calc_ebitda = calc_rev - opex_fixed
        if calc_b2c != b2c or calc_rev != rev or calc_ebitda != ebitda:
            formula_ok = False

    rows.append(
        (
            "B2C чувствит.: монотонность",
            "проверок ↑ → EBITDA ↑",
            f"{len(B2C_CONV_SENSITIVITY)} строк",
            _status(mono_ok),
        )
    )
    rows.append(
        (
            "B2C чувствит.: формула",
            f"B2B {b2b_fixed} + B2C + API {api_fixed} − OPEX {opex_fixed}",
            "фикс. B2B 8%, меняется только B2C",
            _status(formula_ok),
        )
    )

    # --- Базовый план = строка 0,232% в чувствительности ---
    base_row = B2C_CONV_SENSITIVITY[1]
    base_ok = (
        base_row[1].replace(" ", "") == str(FIN_Y3["b2c_checks"])
        and int(base_row[2]) == FIN_Y3["b2c_k"]
        and int(base_row[4]) == ebitda_k(FIN_Y3)
    )
    rows.append(
        (
            "B2C 2029 = строка чувствит.",
            f"{base_row[1]} проверок, EBITDA {base_row[4]}",
            f"FIN_Y3: {FIN_Y3['b2c_checks']:,}, EBITDA {ebitda_k(FIN_Y3)}".replace(",", " "),
            _status(base_ok),
        )
    )

    # --- B2B sensitivity sample ---
    b2b_8 = ebitda_sensitivity_b2b_share(0.08)
    rows.append(
        (
            "B2B 8% = EBITDA осторожн.",
            str(b2b_8),
            f"совпадает с FIN_Y3 {ebitda_k(FIN_Y3)} при B2C фикс.",
            _status(b2b_8 == ebitda_k(FIN_Y3)),
        )
    )

    return rows


def assert_financial_model() -> None:
    """Выбросить AssertionError при расхождении (для build_konkurs_pdf)."""
    bad = [r for r in run_financial_audit() if r[3] != "OK"]
    if bad:
        lines = "\n".join(f"  {r[0]}: {r[2]} → {r[1]} ({r[3]})" for r in bad)
        raise AssertionError(f"Financial model audit failed:\n{lines}")
