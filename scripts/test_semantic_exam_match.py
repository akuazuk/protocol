#!/usr/bin/env python3
"""Э2 offline-тест: улучшение матча exam/treatment за счёт каталога синонимов.

Сравнивает semantic_presence_check БЕЗ каталога (только старый _ALIAS_MAP) и С каталогом
на реалистичных парах «термин протокола» ↔ «текст КЗ с сокращениями».

  python3 scripts/test_semantic_exam_match.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "clinical_knowledge"


def _load():
    pkg = types.ModuleType("clinical_knowledge")
    pkg.__path__ = [str(CK)]  # type: ignore[attr-defined]
    sys.modules["clinical_knowledge"] = pkg
    for name in ("term_catalog", "semantic_rule_fallback"):
        spec = importlib.util.spec_from_file_location(f"clinical_knowledge.{name}", CK / f"{name}.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"clinical_knowledge.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["clinical_knowledge.semantic_rule_fallback"], sys.modules["clinical_knowledge.term_catalog"]


# (термин протокола, текст КЗ, ожидается ли совпадение)
CASES = [
    ("Общий анализ крови развернутый", "Выполнен ОАК: гемоглобин 132, СОЭ 12.", True),
    ("Ультразвуковое исследование органов брюшной полости", "По данным УЗИ ОБП патологии не выявлено.", True),
    ("Эзофагогастродуоденоскопия", "Проведена ФГДС: эрозивный гастрит.", True),
    ("Электрокардиография", "ЭКГ: синусовый ритм, ЧСС 72.", True),
    ("Магнитно-резонансная томография", "Рекомендовано МРТ головного мозга.", True),
    ("Биохимический анализ крови", "БАК в пределах нормы.", True),
    ("Тиреотропный гормон", "ТТГ 2.1 мкМЕ/мл.", True),
    ("Компьютерная томография", "Выполнена РКТ органов грудной клетки.", True),
    ("Холтеровское мониторирование", "Рекомендован холтер ЭКГ на 24 часа.", True),
    ("Электронейромиография", "Назначена ЭНМГ нижних конечностей.", True),
    ("Ультразвуковая допплерография", "УЗДГ вен нижних конечностей.", True),
    ("ингибиторы протонной помпы", "Назначен омепразол 20 мг 2 раза в день.", True),
    ("нестероидные противовоспалительные средства", "Диклофенак 75 мг в/м при боли.", True),
    ("статины", "Рекомендован аторвастатин 20 мг на ночь.", True),
    ("антикоагулянты", "Продолжить ривароксабан 20 мг.", True),
    # негативы: не должно матчиться
    ("Колоноскопия", "Жалобы на сухой кашель и насморк.", False),
    ("Спермограмма", "Осмотр глаз: острота зрения 1.0.", False),
]


def run(srf, use_catalog: bool) -> tuple[int, int, list[str]]:
    orig = srf._catalog_aliases
    if not use_catalog:
        srf._catalog_aliases = lambda term: []  # type: ignore[assignment]
    tp = fp = 0
    misses: list[str] = []
    try:
        for term, text, expect in CASES:
            res = srf.semantic_presence_check(text, term)
            matched = bool(res.get("matched"))
            if expect and matched:
                tp += 1
            elif expect and not matched:
                misses.append(f"MISS: «{term}» в «{text[:40]}…»")
            elif (not expect) and matched:
                fp += 1
                misses.append(f"FALSE+: «{term}» в «{text[:40]}…» ({res.get('method')})")
    finally:
        srf._catalog_aliases = orig  # type: ignore[assignment]
    return tp, fp, misses


def main() -> int:
    srf, tc = _load()
    pos = sum(1 for *_, e in CASES if e)
    neg = len(CASES) - pos
    print(f"catalog_available={tc.catalog_available()}  (exams+drugs indexed)")
    print(f"кейсов: {len(CASES)} (позитивных {pos}, негативных {neg})\n")

    b_tp, b_fp, _ = run(srf, use_catalog=False)
    a_tp, a_fp, a_miss = run(srf, use_catalog=True)

    print(f"БЕЗ каталога:  recall {b_tp}/{pos} = {b_tp/pos:.0%}   false+ {b_fp}/{neg}")
    print(f"С каталогом:   recall {a_tp}/{pos} = {a_tp/pos:.0%}   false+ {a_fp}/{neg}")
    print(f"Прирост recall: +{a_tp - b_tp} кейсов")
    if a_miss:
        print("\nОстаток:")
        for m in a_miss:
            print("  -", m)
    ok = a_tp >= b_tp and a_fp == 0 and a_tp / pos >= 0.9
    print("\nИТОГ:", "OK" if ok else "ТРЕБУЕТ ВНИМАНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
