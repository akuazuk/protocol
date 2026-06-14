#!/usr/bin/env python3
"""Генерация tests/fixtures/search_methodist_probe.jsonl (100+ кейсов)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "search_methodist_probe.jsonl"

# (id_suffix, group, query, population, icd, slugs, expected, reject)
_ROWS: list[tuple] = [
    # URI / pulmonology (14)
    ("01", "uri_adult", "сухой кашель и температура 38", "adult", None, None, ["бронхит", "респиратор", "орви"], ["саркоид", "паллиат"]),
    ("02", "uri_adult", "сухой кашель и температура 38", "adult", ["R05"], None, ["бронхит", "респиратор"], ["саркоид"]),
    ("03", "uri_adult", "болит горло и трудно глотать", "adult", ["R07.0"], None, ["оторin", "лор", "фаринг"], ["анестез", "гэрб"]),
    ("04", "uri_adult", "кашель и температура 39", "adult", None, None, ["pulmonolog", "бронхит", "орви"], ["саркоид"]),
    ("05", "uri_adult", "насморк кашель слабость 5 дней", "adult", None, None, ["орви", "респиратор"], ["трансплант"]),
    ("06", "uri_adult", "J06.9 ОРВИ насморк кашель", "adult", ["J06.9"], None, ["орви", "респиратор"], []),
    ("07", "uri_adult", "J18.9 пневмония кашель одышка", "adult", ["J18.9"], None, ["пневмон", "pulmonolog"], []),
    ("08", "uri_adult", "J45.9 бронхиальная астма приступ", "adult", ["J45.9"], None, ["астм", "pulmonolog"], []),
    ("09", "uri_pediatric", "кашель и температура у ребёнка 4 года", "pediatric", None, None, ["pediatr", "дет", "орви"], []),
    ("10", "uri_pediatric", "кашель", "pediatric", None, None, ["pediatr", "орви", "респиратор"], []),
    ("11", "uri_pediatric", "ORVI у ребенка 2 года насморк", "pediatric", None, None, ["pediatr", "орви", "дет"], []),
    ("12", "pulmonology", "острый бронхит кашель", "adult", None, None, ["бронхит", "pulmonolog"], []),
    ("13", "pulmonology", "пневмония внебольничная", "adult", None, None, ["пневмон", "pulmonolog"], []),
    ("14", "pulmonology", "J20.9 острый бронхит", "adult", ["J20.9"], None, ["бронхит"], []),
    # ENT (8)
    ("15", "ent", "ангина гнойная боль в горле", "adult", None, None, ["оториноларинг", "фаринг", "ангин"], []),
    ("16", "ent", "J02.9 фарингит", "adult", ["J02.9"], None, ["фаринг", "оторin", "лор"], []),
    ("17", "ent", "J03.9 тонзиллит", "adult", ["J03.9"], None, ["оториноларинг", "тонзилл", "ангин"], []),
    ("18", "ent", "отит средний боль в ухе", "adult", None, None, ["отит", "оториноларинг"], []),
    ("19", "ent", "H66.9 отит", "adult", ["H66.9"], None, ["отит", "оториноларинг"], []),
    ("20", "ent_allergy", "аллергический ринит зуд чихание", "adult", None, None, ["ринит", "allergolog", "оториноларинг"], ["буллез", "дерматит"]),
    ("21", "ent_allergy", "J30.4 аллергический ринит", "adult", ["J30.4"], None, ["ринит", "allerg"], ["буллез"]),
    ("22", "ent", "хронический риносинусит гнойные выделения", "adult", None, None, ["риносинус", "sinus", "otorin"], []),
    # Cardiology (10)
    ("23", "cardiology", "гипертоническая болезнь давление 170/100", "adult", None, None, ["гипертенз", "krovoobrash"], ["варикоз", "вен "]),
    ("24", "cardiology", "I10 эссенциальная гипертензия", "adult", ["I10"], ["bolezni-sistemy-krovoobrashcheniya"], ["гипертенз"], []),
    ("25", "cardiology", "I21 инфаркт миокарда", "adult", ["I21"], None, ["инфаркт", "krovoobrash"], []),
    ("26", "cardiology", "I50.9 сердечная недостаточность одышка", "adult", ["I50.9"], None, ["сердечн", "недостат"], []),
    ("27", "cardiology", "I48 фибрилляция предсердий", "adult", ["I48"], None, ["фibrill", "аритм", "krovoobrash"], []),
    ("28", "cardiology", "I20 стенокардия за грудиной", "adult", ["I20"], None, ["стенокард", "krovoobrash"], []),
    ("29", "cardiology", "артериальная гипертензия головная боль", "adult", None, None, ["гипертенз", "krovoobrash"], []),
    ("30", "cardiology", "хроническая сердечная недостаточность отёки", "adult", None, None, ["сердечн", "недостат"], []),
    ("31", "cardiology", "I25 ишемическая болезнь сердца", "adult", ["I25"], None, ["ишем", "krovoobrash", "cardio"], []),
    ("32", "cardiology", "I63 ишемический инсульт", "adult", ["I63"], None, ["инсульт", "nevrolog"], []),
    # Endocrinology (7)
    ("33", "endocrinology", "E11.9 сахарный диабет 2 типа", "adult", ["E11.9"], None, ["диабет", "endokrin"], []),
    ("34", "endocrinology", "E10 сахарный диабет 1 типа", "adult", ["E10"], None, ["диабет", "endokrin"], []),
    ("35", "endocrinology", "E03 гипотиреоз слабость", "adult", ["E03"], None, ["гипотиреоз", "endokrin", "щитовид"], []),
    ("36", "endocrinology", "E05 тиреотоксикоз", "adult", ["E05"], None, ["тиреотокс", "endokrin"], []),
    ("37", "endocrinology", "E66 ожирение", "adult", ["E66"], None, ["ожирен", "endokrin"], []),
    ("38", "endocrinology", "сахарный диабет жажда полиурия", "adult", None, None, ["диабет", "endokrin"], []),
    ("39", "endocrinology", "гипотиреоз АIT", "adult", None, None, ["гипотиреоз", "endokrin"], []),
    # Gastroenterology (10)
    ("40", "gastroenterology", "K29 гастрит изжога", "adult", ["K29"], None, ["гастрит", "gastroenterolog"], []),
    ("41", "gastroenterology", "K85 острый панкреатит", "adult", ["K85"], None, ["панкреатит", "gastro"], []),
    ("42", "gastroenterology", "K80 желчнокаменная болезнь", "adult", ["K80"], None, ["желчн", "gastro"], []),
    ("43", "gastroenterology", "K57 дивертикулез", "adult", ["K57"], None, ["дивертик", "gastro"], []),
    ("44", "gastroenterology", "K50 болезнь Крона", "adult", ["K50"], None, ["крона", "gastro"], []),
    ("45", "gastroenterology", "острая боль в животе тошнота", "adult", None, None, ["gastroenterolog", "khirurg"], []),
    ("46", "gastroenterology", "K21 ГЭРБ изжога", "adult", ["K21"], None, ["рефлюкс", "гэрб", "gastro"], []),
    ("47", "gastroenterology", "K92 желудочное кровотечение", "adult", ["K92"], None, ["gastro", "кровотеч"], []),
    ("48", "gastroenterology", "диспепсия bloating после еды", "adult", None, None, ["dиспепс", "gastro", "гастрит"], []),
    ("49", "gastroenterology", "K74 цирроз печени", "adult", ["K74"], None, ["цирроз", "gastro", "печен"], []),
    # Neurology (8)
    ("50", "neurology", "G43 мигрень с аурой", "adult", ["G43"], None, ["мигрен", "nevrolog"], []),
    ("51", "neurology", "головная боль мигрень", "adult", None, None, ["мигрен", "nevrolog", "головн"], []),
    ("52", "neurology", "G40 эпилепсия приступ", "adult", ["G40"], None, ["эпилеп", "nevrolog"], []),
    ("53", "neurology", "G35 рассеянный склероз", "adult", ["G35"], None, ["склероз", "nevrolog"], []),
    ("54", "neurology", "G20 болезнь Паркинсона", "adult", ["G20"], None, ["паркинсон", "nevrolog"], []),
    ("55", "neurology", "R51 головная боль", "adult", ["R51"], None, ["головн", "nevrolog"], []),
    ("56", "neurology", "ишемический инсульт слабость руки", "adult", None, None, ["инсульт", "nevrolog"], []),
    ("57", "neurology", "G45 ТИА", "adult", ["G45"], None, ["инсульт", "nevrolog", "cerebro"], []),
    # Nephrology / urology (8)
    ("58", "nephrology", "N18 хроническая болезнь почек", "adult", ["N18"], None, ["nefrolog", "почек"], []),
    ("59", "nephrology", "N17 острое повреждение почек", "adult", ["N17"], None, ["nefrolog", "почек"], []),
    ("60", "urology", "N30 цистит дизурия", "adult", ["N30"], None, ["цистит", "urolog"], []),
    ("61", "urology", "N20 мочекаменная болезнь", "adult", ["N20"], None, ["мочекам", "urolog"], []),
    ("62", "urology", "N40 аденома простаты", "adult", ["N40"], None, ["простат", "urolog"], []),
    ("63", "urology", "цистит жжение при мочеиспускании", "adult", None, None, ["цистит", "urolog"], []),
    ("64", "urology", "N39 инфекция мочевых путей", "adult", ["N39"], None, ["urolog", "цистит", "мочев"], []),
    ("65", "nephrology", "N10 острый пиелонефрит", "adult", ["N10"], None, ["пиелонеф", "nefrolog"], []),
    # Rheumatology / orthopedics (7)
    ("66", "rheumatology", "M32.9 системная красная волчанка", "adult", ["M32.9"], None, ["revmatolog", "волчан"], ["буллез"]),
    ("67", "rheumatology", "красная волчанка суставы сыпь", "adult", None, None, ["revmatolog", "волчан"], ["буллез"]),
    ("68", "rheumatology", "M05 ревматоидный артрит", "adult", ["M05"], None, ["revmatolog", "артрит"], []),
    ("69", "rheumatology", "M17 гонартроз", "adult", ["M17"], None, ["артроз", "revmatolog", "ortoped"], []),
    ("70", "rheumatology", "M54 радикулопатия поясница", "adult", ["M54"], None, ["позвон", "nevrolog", "revmatolog"], []),
    ("71", "rheumatology", "M79 фибromyalgia миalgia", "adult", ["M79"], None, ["revmatolog", "nevrolog"], []),
    ("72", "rheumatology", "M10 подагра", "adult", ["M10"], None, ["подагр", "revmatolog"], []),
    # Psychiatry (6)
    ("73", "psychiatry", "F32 депрессивный эпизод", "adult", ["F32"], None, ["psikhiatr", "депресс"], []),
    ("74", "psychiatry", "F41 тревожное расстройство", "adult", ["F41"], None, ["psikhiatr", "тревог"], []),
    ("75", "psychiatry", "F20 шизофрения", "adult", ["F20"], None, ["psikhiatr", "шизофрен"], []),
    ("76", "psychiatry", "F10 алкогольная зависимость", "adult", ["F10"], None, ["narkolog", "psikhiatr", "алкогол"], []),
    ("77", "psychiatry", "депрессия апатия сонливость", "adult", None, None, ["psikhiatr", "депресс"], []),
    ("78", "psychiatry", "F31 биполярное расстройство", "adult", ["F31"], None, ["psikhiatr", "аффектив"], []),
    # Obstetrics / gynecology (8)
    ("79", "obstetrics", "беременность 32 недели головная боль отёки", "pregnant", None, None, ["akusher", "беремен", "ginekolog", "родов"], ["нervn", "неврolog", "дет_"]),
    ("80", "obstetrics", "O14 преэклампсия беременная", "pregnant", ["O14"], None, ["akusher", "преэклам", "беремен"], []),
    ("81", "obstetrics", "O80 роды", "pregnant", ["O80"], None, ["akusher", "род"], []),
    ("82", "obstetrics", "N80 миoma матки", "adult", ["N80"], None, ["ginekolog", "akusher", "миом"], []),
    ("83", "obstetrics", "N92 обильные менструации", "adult", ["N92"], None, ["ginekolog", "akusher"], []),
    ("84", "obstetrics", "D25 миома матки", "adult", ["D25"], None, ["ginekolog", "миом"], []),
    ("85", "obstetrics", "O26 ведение беременности", "pregnant", ["O26"], None, ["akusher", "беремен"], []),
    ("86", "obstetrics", "маstит лактация", "adult", None, None, ["akusher", "ginekolog", "m mastit"], []),
    # Surgery / trauma / burn (8)
    ("87", "surgery", "ожог термический рука 2 степени", "adult", None, None, ["ожог", "термическ", "khirurg"], ["урolog"]),
    ("88", "surgery", "T30 ожог кипятком", "adult", ["T30"], None, ["ожог", "термическ"], ["урolog"]),
    ("89", "surgery", "S06 сотрясение мозга", "adult", ["S06"], None, ["череп", "nevrolog", "travm"], []),
    ("90", "surgery", "S72 перелом бедра", "adult", ["S72"], None, ["перелом", "travm", "ortoped"], []),
    ("91", "surgery", "K35 острый аппендицит", "adult", ["K35"], None, ["аппендиц", "khirurg"], []),
    ("92", "surgery", "K40 паховая грыжа", "adult", ["K40"], None, ["грыж", "khirurg"], []),
    ("93", "surgery", "рана брюшной стенки", "adult", None, None, ["ран", "khirurg", "travm"], []),
    ("94", "surgery", "S01 рана головы", "adult", ["S01"], None, ["ран", "travm", "khirurg"], []),
    # Oncology / hematology / infectious / derm / eye / dental (12)
    ("95", "oncology", "C50 рак молочной железы", "adult", ["C50"], None, ["novoobraz", "онкolog", "злокач"], []),
    ("96", "oncology", "C18 рак толстой кишки", "adult", ["C18"], None, ["novoobraz", "онкolog", "кишк"], []),
    ("97", "oncology", "подозрение на злокачественное новообразование", "adult", None, None, ["novoobraz", "онкolog"], []),
    ("98", "hematology", "D50 железодефицитная анемия", "adult", ["D50"], None, ["анем", "gematolog"], []),
    ("99", "hematology", "D64 анемия слабость", "adult", ["D64"], None, ["анем", "gematolog"], []),
    ("100", "infectious", "A09 острая диарея", "adult", ["A09"], None, ["infektsion", "диар"], []),
    ("101", "infectious", "B20 ВИЧ инфекция", "adult", ["B20"], None, ["vich", "infektsion"], []),
    ("102", "dermatology", "L20 атопический дерматит", "adult", ["L20"], None, ["dermat", "дерматит"], []),
    ("103", "dermatology", "L40 псoriasis", "adult", ["L40"], None, ["psorias", "dermat"], []),
    ("104", "ophthalmology", "H10 конъюнктивит", "adult", ["H10"], None, ["oftalm", "конъюнктив"], []),
    ("105", "ophthalmology", "H40 глаукома", "adult", ["H40"], None, ["глаук", "oftalm"], []),
    ("106", "stomatology", "K02 кариес зуб", "adult", ["K02"], None, ["stomat", "кaries", "зуб"], []),
    ("107", "stomatology", "K04 пulpit зуб", "adult", ["K04"], None, ["stomat", "пulpit", "зуб"], []),
    ("108", "mixed", "кашель бронхит J20", "adult", None, None, ["бронхит", "pulmonolog"], []),
    ("109", "mixed", "E11 полиурия жажда", "adult", None, None, ["диабет", "endokrin"], []),
    ("110", "mixed", "N39.0 цистит дизурия", "adult", None, None, ["цистит", "urolog"], []),
]


def main() -> int:
    lines: list[str] = []
    for suffix, group, query, pop, icd, slugs, expected, reject in _ROWS:
        row: dict = {
            "id": f"probe{suffix}",
            "group": group,
            "query": query,
            "population": pop,
            "expected_contains": expected,
            "reject_contains": reject,
        }
        if icd:
            row["icd_codes"] = icd
        if slugs:
            row["category_slugs"] = slugs
        lines.append(json.dumps(row, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} probes → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
