#!/usr/bin/env python3
"""Прогон 100 симптомных запросов: шаг 2 МКБ (+ опционально шаг 4 протоколы).

Пример:
  python3 scripts/run_symptom_icd_probe.py --base https://protocol-bimy.onrender.com
  python3 scripts/run_symptom_icd_probe.py --local --no-gemini
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "symptom_icd_probe_100.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "reports" / "symptom_icd_probe_latest.jsonl"
DEFAULT_MD = ROOT / "data" / "ml" / "reports" / "symptom_icd_probe_latest.md"

_TRAUMA_MARKERS = re.compile(
    r"травм|перелом|ушиб|рана|ожог|отравлен|инородн|авар|дтп|падени|удар",
    re.I,
)
_EXOTIC_A = re.compile(r"^A(2[0-5]|26|27|28|3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9])", re.I)

# 100 симптомных жалоб без явного кода МКБ в тексте
PROBE_ROWS: list[dict[str, Any]] = [
    # URI / pulmonology (12)
    {"id": "s01", "group": "uri", "query": "сухой кашель и температура 38", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y", "Z"]},
    {"id": "s02", "group": "uri", "query": "кашель и температура 39", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s03", "group": "uri", "query": "насморк кашель слабость 5 дней", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s04", "group": "uri", "query": "болит горло и трудно глотать", "population": "adult", "expected_prefixes": ["J", "R", "H"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s05", "group": "uri", "query": "кашель с мокротой одышка", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s06", "group": "uri", "query": "острый бронхит кашель", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s07", "group": "uri", "query": "пневмония кашель одышка", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s08", "group": "uri_ped", "query": "кашель и температура у ребёнка 4 года", "population": "pediatric", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s09", "group": "uri_ped", "query": "ORVI у ребенка 2 года насморк", "population": "pediatric", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s10", "group": "uri", "query": "хрипы в груди кашель ночью", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s11", "group": "uri", "query": "бронхиальная астма приступ удушье", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s12", "group": "uri", "query": "лихорадка 39 сухой кашель", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # ENT (8)
    {"id": "s13", "group": "ent", "query": "ангина гнойная боль в горле", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s14", "group": "ent", "query": "отит средний боль в ухе", "population": "adult", "expected_prefixes": ["H", "J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s15", "group": "ent", "query": "хронический риносинусит гнойные выделения", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s16", "group": "ent", "query": "заложенность носа головная боль", "population": "adult", "expected_prefixes": ["J", "R", "G"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s17", "group": "ent", "query": "фарингит першение в горле", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s18", "group": "ent", "query": "аденоидит у ребенка храп", "population": "pediatric", "expected_prefixes": ["J", "R", "H"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s19", "group": "ent", "query": "ларингит охриплость голоса", "population": "adult", "expected_prefixes": ["J", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s20", "group": "allergy", "query": "аллергический ринит зуд чихание", "population": "adult", "expected_prefixes": ["J", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Cardiology (10)
    {"id": "s21", "group": "cardio", "query": "гипертоническая болезнь давление 170/100", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s22", "group": "cardio", "query": "боль за грудиной при нагрузке", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s23", "group": "cardio", "query": "аритмия перебои в сердце", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s24", "group": "cardio", "query": "сердечная недостаточность отёки ног", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s25", "group": "cardio", "query": "артериальная гипертензия головная боль", "population": "adult", "expected_prefixes": ["I", "R", "G"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s26", "group": "cardio", "query": "острая боль в груди и одышка", "population": "emergency", "expected_prefixes": ["I", "R", "J"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s27", "group": "cardio", "query": "тахикардия сердцебиение в покое", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s28", "group": "cardio", "query": "ишемическая болезнь сердца стенокардия", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s29", "group": "cardio", "query": "инфаркт миокарда боль в груди", "population": "emergency", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s30", "group": "cardio", "query": "варикозное расширение вен ног", "population": "adult", "expected_prefixes": ["I", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Endocrinology (6)
    {"id": "s31", "group": "endo", "query": "сахарный диабет жажда полиурия", "population": "adult", "expected_prefixes": ["E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s32", "group": "endo", "query": "гипотиреоз слабость набор веса", "population": "adult", "expected_prefixes": ["E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s33", "group": "endo", "query": "тиреотоксикоз тремор потливость", "population": "adult", "expected_prefixes": ["E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s34", "group": "endo", "query": "ожирение индекс массы тела 35", "population": "adult", "expected_prefixes": ["E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s35", "group": "endo", "query": "гестационный диабет беременность", "population": "pregnant", "expected_prefixes": ["E", "O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s36", "group": "endo", "query": "гипогликемия потливость дрожь", "population": "adult", "expected_prefixes": ["E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Gastroenterology (12)
    {"id": "s37", "group": "gi", "query": "гастрит изжога после еды", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s38", "group": "gi", "query": "острая боль в животе тошнота", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s39", "group": "gi", "query": "диспепсия вздутие после еды", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s40", "group": "gi", "query": "кровь в кале и шишки воспаленные в заднем проходе", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y", "Z"], "expected_top": ["K64", "K62", "K92"]},
    {"id": "s41", "group": "gi", "query": "запор боль в животе 4 дня", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s42", "group": "gi", "query": "понос рвота подозрение на кишечную инфекцию", "population": "adult", "expected_prefixes": ["K", "A", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s43", "group": "gi", "query": "болезнь Крона диарея кровь", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s44", "group": "gi", "query": "цирроз печени асцит", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s45", "group": "gi", "query": "желчнокаменная болезнь боль в правом подреберье", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s46", "group": "gi", "query": "острый панкреатит опоясывающая боль", "population": "emergency", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s47", "group": "gi", "query": "рефлюкс изжога ночью", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s48", "group": "gi", "query": "геморрой кровотечение при дефекации", "population": "adult", "expected_prefixes": ["K", "R"], "bad_prefixes": ["T", "X", "Y", "Z"]},
    # Neurology (8)
    {"id": "s49", "group": "neuro", "query": "головная боль мигрень с аурой", "population": "adult", "expected_prefixes": ["G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s50", "group": "neuro", "query": "эпилепсия судорожный приступ", "population": "adult", "expected_prefixes": ["G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s51", "group": "neuro", "query": "ишемический инсульт слабость руки", "population": "emergency", "expected_prefixes": ["I", "G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s52", "group": "neuro", "query": "головокружение шум в ушах", "population": "adult", "expected_prefixes": ["G", "H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s53", "group": "neuro", "query": "болезнь Паркинсона тремор", "population": "adult", "expected_prefixes": ["G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s54", "group": "neuro", "query": "радикулит боль в пояснице отдаёт в ногу", "population": "adult", "expected_prefixes": ["M", "G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s55", "group": "neuro", "query": "онемение пальцев рук ночью", "population": "adult", "expected_prefixes": ["G", "M", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s56", "group": "neuro", "query": "рассеянный склероз нарушение зрения", "population": "adult", "expected_prefixes": ["G", "H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Nephrology / urology (8)
    {"id": "s57", "group": "uro", "query": "цистит жжение при мочеиспускании", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s58", "group": "uro", "query": "мочекаменная болезнь колика в пояснице", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s59", "group": "uro", "query": "аденома простаты затруднённое мочеиспускание", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s60", "group": "nephro", "query": "хроническая болезнь почек отёки", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s61", "group": "nephro", "query": "пиелонефрит боль в пояснице лихорадка", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s62", "group": "uro", "query": "инфекция мочевых путей у женщины", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s63", "group": "uro", "query": "гематурия без боли", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s64", "group": "nephro", "query": "нефротический синдром отёки белок в моче", "population": "adult", "expected_prefixes": ["N", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Rheumatology / orthopedics (7)
    {"id": "s65", "group": "rheum", "query": "ревматоидный артрит боль в суставах", "population": "adult", "expected_prefixes": ["M", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s66", "group": "rheum", "query": "системная красная волчанка сыпь суставы", "population": "adult", "expected_prefixes": ["M", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s67", "group": "rheum", "query": "гонартроз боль в колене при ходьбе", "population": "adult", "expected_prefixes": ["M", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s68", "group": "rheum", "query": "подагра приступ в большом пальце стопы", "population": "adult", "expected_prefixes": ["M", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s69", "group": "ortho", "query": "перелом лучевой кости после падения", "population": "emergency", "expected_prefixes": ["S", "M", "R"], "bad_prefixes": []},
    {"id": "s70", "group": "ortho", "query": "ушиб колена отёк гематома", "population": "adult", "expected_prefixes": ["S", "M", "R"], "bad_prefixes": []},
    {"id": "s71", "group": "rheum", "query": "остеопороз перелом позвонка", "population": "adult", "expected_prefixes": ["M", "S", "R"], "bad_prefixes": []},
    # Psychiatry (5)
    {"id": "s72", "group": "psych", "query": "депрессия апатия сонливость", "population": "adult", "expected_prefixes": ["F", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s73", "group": "psych", "query": "тревожное расстройство панические атаки", "population": "adult", "expected_prefixes": ["F", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s74", "group": "psych", "query": "бессонница хроническая", "population": "adult", "expected_prefixes": ["F", "G", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s75", "group": "psych", "query": "алкогольная зависимость абстиненция", "population": "adult", "expected_prefixes": ["F", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s76", "group": "psych", "query": "биполярное расстройство мания", "population": "adult", "expected_prefixes": ["F", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # OB/GYN (8)
    {"id": "s77", "group": "obgyn", "query": "беременность тошнота рвота", "population": "pregnant", "expected_prefixes": ["O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s78", "group": "obgyn", "query": "угроза прерывания беременности кровотечение", "population": "pregnant", "expected_prefixes": ["O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s79", "group": "obgyn", "query": "эндометриоз боли при менструации", "population": "adult", "expected_prefixes": ["N", "O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s80", "group": "obgyn", "query": "миома матки обильные менструации", "population": "adult", "expected_prefixes": ["D", "N", "O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s81", "group": "obgyn", "query": "вульвовагинит зуд выделения", "population": "adult", "expected_prefixes": ["N", "O", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s82", "group": "obgyn", "query": "климакс приливы потливость", "population": "adult", "expected_prefixes": ["N", "E", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s83", "group": "obgyn", "query": "мастит лактация лихорадка", "population": "adult", "expected_prefixes": ["N", "O", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s84", "group": "obgyn", "query": "послеродовое кровотечение", "population": "emergency", "expected_prefixes": ["O", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Dermatology (6)
    {"id": "s85", "group": "derm", "query": "атопический дерматит зуд кожи", "population": "adult", "expected_prefixes": ["L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s86", "group": "derm", "query": "псориаз бляшки на локтях", "population": "adult", "expected_prefixes": ["L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s87", "group": "derm", "query": "крапивница сыпь после еды", "population": "adult", "expected_prefixes": ["L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s88", "group": "derm", "query": "герпес на губе пузырьки", "population": "adult", "expected_prefixes": ["B", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s89", "group": "derm", "query": "грибок ногтей утолщение", "population": "adult", "expected_prefixes": ["B", "L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s90", "group": "derm", "query": "акне воспалённые элементы на лице", "population": "adult", "expected_prefixes": ["L", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Ophthalmology (4)
    {"id": "s91", "group": "ophth", "query": "конъюнктивит покраснение глаза", "population": "adult", "expected_prefixes": ["H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s92", "group": "ophth", "query": "глаукома снижение зрения", "population": "adult", "expected_prefixes": ["H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s93", "group": "ophth", "query": "катаракта помутнение хрусталика", "population": "adult", "expected_prefixes": ["H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s94", "group": "ophth", "query": "боль в глазу и светобоязнь", "population": "adult", "expected_prefixes": ["H", "R"], "bad_prefixes": ["T", "X", "Y"]},
    # Emergency / poison (4)
    {"id": "s95", "group": "emergency", "query": "отравление угарным газом головная боль", "population": "emergency", "expected_prefixes": ["T", "X", "R"], "bad_prefixes": []},
    {"id": "s96", "group": "emergency", "query": "ожог руки кипятком", "population": "emergency", "expected_prefixes": ["T", "R"], "bad_prefixes": []},
    {"id": "s97", "group": "emergency", "query": "инородное тело в дыхательных путях", "population": "emergency", "expected_prefixes": ["T", "W", "R"], "bad_prefixes": []},
    {"id": "s98", "group": "emergency", "query": "анафилаксия после укуса пчелы", "population": "emergency", "expected_prefixes": ["T", "L", "R"], "bad_prefixes": []},
    # Misc (2)
    {"id": "s99", "group": "hematology", "query": "анемия слабость бледность", "population": "adult", "expected_prefixes": ["D", "R"], "bad_prefixes": ["T", "X", "Y"]},
    {"id": "s100", "group": "oncology", "query": "подозрение на опухоль лёгкого кашель кровь", "population": "adult", "expected_prefixes": ["C", "J", "R"], "bad_prefixes": ["T", "X", "Y"]},
]

_POP = {
    "adult": "Контекст подбора: взрослое население",
    "pediatric": "Контекст подбора: детское население",
    "pregnant": "Контекст подбора: беременные",
    "emergency": "Контекст подбора: неотложная помощь",
}


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in PROBE_ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        _write_fixture(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _build_query(row: dict[str, Any]) -> str:
    parts = [str(row.get("query") or "").strip()]
    pop = str(row.get("population") or "").strip().lower()
    if pop in _POP:
        parts.append(_POP[pop])
    return "\n".join(p for p in parts if p)


def _code_prefix(code: str) -> str:
    return (code or "").strip().upper()[:1]


def _code_starts(code: str, prefix: str) -> bool:
    return (code or "").upper().startswith(prefix.upper())


_ALLOWED_T78_GROUPS = frozenset({"derm", "allergy", "emergency"})


def _code_allowed_exception(code: str, group: str) -> bool:
    cu = (code or "").upper()
    if cu.startswith("T78") and group in _ALLOWED_T78_GROUPS:
        return True
    return False


def _score_icd(codes: list[str], row: dict[str, Any]) -> dict[str, Any]:
    expected = list(row.get("expected_prefixes") or [])
    bad = list(row.get("bad_prefixes") or [])
    expected_top = list(row.get("expected_top") or [])
    group = str(row.get("group") or "")
    trauma = bool(_TRAUMA_MARKERS.search(str(row.get("query") or "")))
    top1 = codes[0] if codes else ""
    top3 = codes[:3]
    top4 = codes[:4]

    def prefix_ok(c: str) -> bool:
        if not c:
            return False
        if _code_allowed_exception(c, group):
            return True
        if expected_top and any(_code_starts(c, p) for p in expected_top):
            return True
        return _code_prefix(c) in expected

    top1_ok = prefix_ok(top1)
    top3_ok = any(prefix_ok(c) for c in top3)
    bad_hits = [
        c for c in top3
        if _code_prefix(c) in bad and not trauma and not _code_allowed_exception(c, group)
    ]
    top1_bad = bool(
        top1 and _code_prefix(top1) in bad and not trauma and not _code_allowed_exception(top1, group)
    )
    exotic = [c for c in top4 if _EXOTIC_A.match(c or "")]
    empty = not codes
    verdict = "ok"
    if empty:
        verdict = "empty"
    elif top1_bad:
        verdict = "bad_prefix"
    elif exotic and group in ("uri", "uri_ped", "gi", "ent"):
        verdict = "exotic_fever"
    elif not top1_ok:
        verdict = "top1_miss"
    elif not top3_ok:
        verdict = "top3_miss"
    return {
        "top1": top1,
        "top3": top3,
        "top1_ok": top1_ok,
        "top3_ok": top3_ok,
        "bad_hits": bad_hits,
        "exotic_top4": exotic,
        "verdict": verdict,
        "empty": empty,
    }


def _extract_codes_from_funnel(body: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for ch in body.get("choices") or []:
        if isinstance(ch, dict):
            c = str(ch.get("id") or "").strip()
            if c and c not in seen:
                seen.add(c)
                codes.append(c)
    icd = body.get("icd") or {}
    for bucket in ("detected", "suggested"):
        for row in icd.get(bucket) or []:
            if isinstance(row, dict):
                c = str(row.get("code") or "").strip()
                if c and c not in seen:
                    seen.add(c)
                    codes.append(c)
    for c in icd.get("codes_for_retrieval") or []:
        cs = str(c).strip()
        if cs and cs not in seen:
            seen.add(cs)
            codes.append(cs)
    return codes


_GROUP_PROTOCOL_HINTS: dict[str, list[str]] = {
    "uri": ["орви", "бронхит", "респиратор", "pulmonolog", "оторин"],
    "uri_ped": ["pediatr", "дет", "орви", "респиратор"],
    "ent": ["оторин", "лор", "фаринг", "риносинус", "отит"],
    "allergy": ["allerg", "ринит", "крапивниц"],
    "cardio": ["krovoobrash", "гипертенз", "сердечн", "ишем"],
    "endo": ["endokrin", "диабет", "тирео", "щитовид"],
    "gi": ["gastroenterolog", "гастрит", "кишеч", "панкреат", "гэрб"],
    "neuro": ["nevrolog", "мигрен", "инсульт", "эпилеп", "паркинсон"],
    "nephro": ["nefrolog", "почек"],
    "uro": ["urolog", "цистит", "мочев", "простат"],
    "rheum": ["revmatolog", "артрит", "артроз", "волчан", "подагр"],
    "ortho": ["ortoped", "перелом", "травм"],
    "psych": ["psikhiatr", "депресс", "тревог", "narkolog"],
    "obgyn": ["akusherstvo", "ginekolog", "беремен", "акушер"],
    "derm": ["dermat", "дермат", "псориаз", "атопическ"],
    "ophth": ["oftalmolog", "офтальм", "глауком", "конъюнктив"],
    "emergency": ["неотлож", "экстрен", "anestez", "травм", "ожог"],
    "hematology": ["гематолог", "анеми"],
    "oncology": ["онколог", "novoobraz", "pulmonolog"],
}


def _ensure_rag_loaded() -> None:
    sys.path.insert(0, str(ROOT))
    import rag_server as rs

    if rs._chunks_load_done.is_set():
        rs._require_rag_loaded()
        return
    print("Загрузка RAG (один раз)…", flush=True)
    t0 = time.perf_counter()
    rs._run_load_data_background()
    rs._require_rag_loaded()
    print(f"RAG готов за {time.perf_counter() - t0:.1f}s", flush=True)


def _protocol_hit(row: dict[str, Any], top_path: str, top_title: str) -> bool:
    blob = f"{top_path} {top_title}".lower()
    hints = list(row.get("expected_contains") or []) or _GROUP_PROTOCOL_HINTS.get(
        str(row.get("group") or ""), []
    )
    return any(h.lower() in blob for h in hints if h)


def _local_protocol_search(
    query: str,
    *,
    population: str | None,
    icd_codes: list[str],
) -> tuple[dict[str, Any], int]:
    from clinical_knowledge.search_funnel import handle_search_funnel

    ctx = {"population": population, "icd_codes": icd_codes[:5]}
    t0 = time.perf_counter()
    body = handle_search_funnel(
        query=query,
        step=4,
        context=ctx,
        category_slugs=None,
        session_id=None,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    return body, ms


def _enrich_protocol_review(
    row: dict[str, Any],
    q: str,
    assist_payload: dict[str, Any],
) -> dict[str, Any]:
    from clinical_knowledge.methodist_search_ai_review import build_deterministic_search_ai_review

    ai = build_deterministic_search_ai_review(
        {
            "query": q,
            "llm_json": assist_payload.get("llm_json") or {},
            "retrieval": assist_payload.get("retrieval") or [],
            "icd_codes": assist_payload.get("icd_codes") or [],
            "retrieve_only": True,
            "funnel_context": {"population": row.get("population")},
        }
    )
    protos = (assist_payload.get("llm_json") or {}).get("protocols") or []
    top = protos[0] if protos and isinstance(protos[0], dict) else {}
    top_path = str(top.get("path") or "")
    top_title = str(top.get("title") or "")
    rating = int(ai.get("ranking_rating") or ai.get("ai_rating") or 0)
    return {
        "protocol_top1_path": top_path,
        "protocol_top1_title": (top_title or top_path.rsplit("/", 1)[-1])[:120],
        "protocol_top1_confidence": top.get("confidence_score"),
        "protocol_rating": rating,
        "protocol_verdict": ai.get("ranking_verdict") or ai.get("verdict"),
        "protocol_hit": _protocol_hit(row, top_path, top_title),
        "protocol_top1_relevant": ai.get("top1_relevant"),
        "protocol_top3": [
            {
                "path": pr.get("path"),
                "title": pr.get("title"),
                "confidence": pr.get("confidence_score"),
            }
            for pr in protos[:3]
            if isinstance(pr, dict)
        ],
        "protocol_review": ai,
    }


    url = base.rstrip("/") + "/api/search/funnel"
    payload = json.dumps({"query": query, "step": step, "context": context}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, int((time.perf_counter() - t0) * 1000)


def _local_icd(query: str, *, no_gemini: bool) -> tuple[list[str], dict[str, Any], int]:
    import os

    sys.path.insert(0, str(ROOT))
    if no_gemini:
        os.environ["RAG_GEMINI_QUERY_REFINE"] = "0"
        os.environ["RAG_ICD_PRE_RETRIEVE_INFER"] = "0"
    import rag_server as rs

    t0 = time.perf_counter()
    icd_analysis, _, _, _, err = rs._infer_icd_pipeline_from_full_query(
        query,
        rs.get_gemini() if not no_gemini else None,
        skip_query_refine=no_gemini,
        skip_icd_gemini=no_gemini,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    if err:
        return [], {"error": err}, ms
    codes: list[str] = []
    seen: set[str] = set()
    for bucket in ("detected", "suggested"):
        for row in (icd_analysis or {}).get(bucket) or []:
            if isinstance(row, dict):
                c = str(row.get("code") or "").strip()
                if c and c not in seen:
                    seen.add(c)
                    codes.append(c)
    return codes, icd_analysis or {}, ms


def _run_one(
    row: dict[str, Any],
    *,
    base: str | None,
    local: bool,
    no_gemini: bool,
    with_protocols: bool,
) -> dict[str, Any]:
    q = _build_query(row)
    out: dict[str, Any] = {
        "id": row.get("id"),
        "group": row.get("group"),
        "query": q,
        "population": row.get("population"),
    }
    try:
        if local:
            codes, icd_payload, ms = _local_icd(q, no_gemini=no_gemini)
            out["latency_ms"] = ms
            out["icd_codes"] = codes
            out["icd"] = icd_payload
            if with_protocols and codes:
                pbody, pms = _local_protocol_search(
                    q, population=str(row.get("population") or ""), icd_codes=codes
                )
                out["protocol_latency_ms"] = pms
                assist = pbody.get("assist") or {}
                assist["icd_codes"] = codes[:5]
                out.update(_enrich_protocol_review(row, q, assist))
        else:
            body, ms = _remote_funnel(base or "", q, 2, {})
            out["latency_ms"] = ms
            out["funnel_step2"] = body
            codes = _extract_codes_from_funnel(body)
            out["icd_codes"] = codes
            if body.get("error"):
                out["error"] = body["error"]

            if with_protocols and codes and not body.get("error"):
                ctx = {"population": row.get("population"), "icd_codes": codes[:5]}
                pbody, pms = _remote_funnel(base or "", q, 4, ctx)
                out["protocol_latency_ms"] = pms
                assist = pbody.get("assist") or {}
                assist["icd_codes"] = codes[:5]
                out.update(_enrich_protocol_review(row, q, assist))
    except urllib.error.HTTPError as exc:
        out["error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"
        codes = []
    except Exception as exc:
        out["error"] = str(exc)
        codes = []

    score = _score_icd(codes, row)
    out.update(score)
    return out


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    ok = sum(1 for r in results if r.get("verdict") == "ok")
    top1_ok = sum(1 for r in results if r.get("top1_ok"))
    top3_ok = sum(1 for r in results if r.get("top3_ok"))
    bad = sum(1 for r in results if r.get("verdict") == "bad_prefix")
    exotic = sum(1 for r in results if r.get("verdict") == "exotic_fever")
    empty = sum(1 for r in results if r.get("verdict") == "empty")
    top1_miss = sum(1 for r in results if r.get("verdict") == "top1_miss")
    errs = sum(1 for r in results if r.get("error"))
    latencies = [int(r.get("latency_ms") or 0) for r in results if r.get("latency_ms")]
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_verdict = Counter(r.get("verdict") for r in results)
    bad_codes = Counter()
    for r in results:
        by_group[str(r.get("group"))].append(r)
        for c in r.get("bad_hits") or []:
            bad_codes[c] += 1
        for c in r.get("exotic_top4") or []:
            bad_codes[c] += 1

    group_stats = {}
    for g, rows in sorted(by_group.items()):
        group_stats[g] = {
            "n": len(rows),
            "top1_ok_pct": round(100 * sum(1 for r in rows if r.get("top1_ok")) / len(rows), 1),
            "fail": [r["id"] for r in rows if r.get("verdict") != "ok"],
        }

    worst = sorted(
        [r for r in results if r.get("verdict") != "ok"],
        key=lambda r: (r.get("verdict") != "bad_prefix", r.get("verdict") != "exotic_fever"),
    )[:25]

    proto_rows = [r for r in results if r.get("protocol_rating") is not None]
    proto_summary: dict[str, Any] = {}
    if proto_rows:
        proto_summary = {
            "n": len(proto_rows),
            "hit1_pct": round(100 * sum(1 for r in proto_rows if r.get("protocol_hit")) / len(proto_rows), 1),
            "rating_ge4_pct": round(
                100 * sum(1 for r in proto_rows if int(r.get("protocol_rating") or 0) >= 4) / len(proto_rows), 1
            ),
            "avg_rating": round(
                sum(int(r.get("protocol_rating") or 0) for r in proto_rows) / len(proto_rows), 2
            ),
            "worst_protocol": sorted(
                proto_rows,
                key=lambda r: (int(r.get("protocol_rating") or 0), not r.get("protocol_hit")),
            )[:15],
        }

    return {
        "n": n,
        "ok_pct": round(100 * ok / n, 1) if n else 0,
        "top1_ok_pct": round(100 * top1_ok / n, 1) if n else 0,
        "top3_ok_pct": round(100 * top3_ok / n, 1) if n else 0,
        "bad_prefix": bad,
        "exotic_fever": exotic,
        "empty": empty,
        "top1_miss": top1_miss,
        "errors": errs,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
        "verdicts": dict(by_verdict),
        "bad_codes": bad_codes.most_common(15),
        "group_stats": group_stats,
        "worst": worst,
        "protocol": proto_summary,
    }


def _write_report(summary: dict[str, Any], results: list[dict], out_md: Path, meta: dict) -> None:
    lines = [
        "# Symptom ICD probe (100 queries)",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Mode: {meta.get('mode')}",
        f"- BUILD: `{meta.get('build', '?')}`",
        f"- Probes: **{summary['n']}** (errors {summary['errors']})",
        "",
        "## ICD step-2 metrics",
        "",
        f"- Top-1 clinically plausible (prefix/heuristic): **{summary['top1_ok_pct']}%**",
        f"- Top-3 plausible: **{summary['top3_ok_pct']}%**",
        f"- Fully OK verdict: **{summary['ok_pct']}%**",
        f"- Bad prefix (T/X/Y/Z) in top-3: **{summary['bad_prefix']}**",
        f"- Exotic A** fever in top-4 (URI/GI/ENT): **{summary['exotic_fever']}**",
        f"- Empty ICD: **{summary['empty']}**",
        f"- Top-1 miss (plausible in top-3): **{summary['top1_miss']}**",
        f"- Avg latency: **{summary['avg_latency_ms']}** ms · p95 **{summary['p95_latency_ms']}** ms",
        "",
    ]
    if summary.get("protocol"):
        ps = summary["protocol"]
        lines.extend(
            [
                "## Protocol step-4 metrics",
                "",
                f"- Evaluated: **{ps.get('n')}**",
                f"- Top-1 specialty hit (heuristic): **{ps.get('hit1_pct')}%**",
                f"- AI rating ≥4/5: **{ps.get('rating_ge4_pct')}%**",
                f"- Avg AI rating: **{ps.get('avg_rating')}** / 5",
                "",
            ]
        )
    lines.extend(
        [
        "## Verdicts",
        "",
        ]
    )
    for v, cnt in sorted(summary.get("verdicts", {}).items(), key=lambda x: -x[1]):
        lines.append(f"- `{v}`: {cnt}")
    lines.extend(["", "## By group", ""])
    for g, st in summary.get("group_stats", {}).items():
        fails = ", ".join(st.get("fail") or []) or "-"
        lines.append(f"- **{g}** ({st['n']}): top1 {st['top1_ok_pct']}% · fails: {fails}")
    if summary.get("bad_codes"):
        lines.extend(["", "## Recurring bad codes", ""])
        for code, cnt in summary["bad_codes"]:
            lines.append(f"- `{code}`: {cnt}×")
    lines.extend(["", "## Worst cases", "", "| id | group | verdict | top-1 | top-3 | query |", "|----|-------|---------|-------|-------|-------|"])
    for r in summary.get("worst") or []:
        qshort = (r.get("query") or "").split("\n")[0][:50]
        top3s = ", ".join(r.get("top3") or [])[:40]
        lines.append(
            f"| {r.get('id')} | {r.get('group')} | {r.get('verdict')} | {r.get('top1')} | {top3s} | {qshort} |"
        )
    if summary.get("protocol", {}).get("worst_protocol"):
        lines.extend(
            [
                "",
                "## Worst protocol matches",
                "",
                "| id | rating | hit | top-1 protocol | ICD top-1 |",
                "|----|--------|-----|----------------|-----------|",
            ]
        )
        for r in summary["protocol"]["worst_protocol"]:
            lines.append(
                f"| {r.get('id')} | {r.get('protocol_rating')} | {r.get('protocol_hit')} | "
                f"{(r.get('protocol_top1_title') or '')[:40]} | {r.get('top1')} |"
            )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://protocol-bimy.onrender.com")
    ap.add_argument("--local", action="store_true", help="Локальный _infer_icd_pipeline (без загрузки RAG)")
    ap.add_argument("--no-gemini", action="store_true", help="Только лексикон + hints")
    ap.add_argument("--with-protocols", action="store_true", help="Шаг 4: поиск протоколов + AI-оценка")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = ap.parse_args()

    rows = _load_rows(args.fixture)
    if args.limit > 0:
        rows = rows[: args.limit]

    build = "?"
    if not args.local:
        try:
            with urllib.request.urlopen(args.base.rstrip("/") + "/api/version", timeout=30) as resp:
                build = json.loads(resp.read()).get("version") or build
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    mode = "local/no-gemini" if args.local and args.no_gemini else ("local/full" if args.local else f"remote:{args.base}")
    if args.with_protocols and args.local:
        _ensure_rag_loaded()
    print(f"Probe {len(rows)} queries · mode={mode} · build={build}", flush=True)

    for i, row in enumerate(rows, 1):
        r = _run_one(
            row,
            base=None if args.local else args.base,
            local=args.local,
            no_gemini=args.no_gemini,
            with_protocols=args.with_protocols,
        )
        results.append(r)
        mark = "OK" if r.get("verdict") == "ok" else r.get("verdict")
        extra = ""
        if args.with_protocols and r.get("protocol_rating") is not None:
            extra = f" prot={r.get('protocol_rating')}/5 hit={r.get('protocol_hit')}"
        print(
            f"  [{i:3d}/{len(rows)}] {row.get('id')} {mark:12s} top1={r.get('top1','')} {r.get('latency_ms',0)}ms{extra}",
            flush=True,
        )
        if not args.local:
            time.sleep(0.15)

    summary = _summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"mode": mode, "build": build}
    _write_report(summary, results, args.md, meta)

    snap = ROOT / "data" / "ml" / "symptom_icd_probe_snapshot.json"
    snap.write_text(
        json.dumps({"summary": summary, "meta": meta, "generated": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone: top1={summary['top1_ok_pct']}% ok={summary['ok_pct']}% bad={summary['bad_prefix']} exotic={summary['exotic_fever']}")
    print(f"Report: {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
