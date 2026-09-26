"""Умный поиск по случаям МО: план запроса, SQL-фрагменты, Python-предикат, словарь алиасов."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_search as ms

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_and_tokens() -> None:
    assert ms.normalize("  Гипертоническая, Болезнь!  ") == "гипертоническая болезнь"
    assert ms.normalize("ёлка Ёж") == "елка еж"
    assert ms.tokens("остеохондроз поясничного отдела; I10.5") == ["остеохондроз", "поясничного", "отдела", "i10", "5"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("гипертония", "гипертон"),
        ("гипертензия", "гипертенз"),
        ("остеохондроза", "остеохондроз"),
        ("поясничного", "поясничн"),
        ("острая", "остр"),
        ("миопия", "миоп"),
        ("зоб", "зоб"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert ms.stem(word) == expected


@pytest.mark.parametrize(
    ("q", "prefixes"),
    [
        ("I10", ["I10"]),
        ("i10.", ["I10"]),
        ("I1", ["I1"]),
        ("I10.5", ["I10.5"]),
        ("I10-I15", ["I10", "I11", "I12", "I13", "I14", "I15"]),
        ("К29.7", ["K29.7"]),  # кириллическая К с русской раскладки
        ("гипертония", []),
        ("12345", []),
    ],
)
def test_icd_prefixes(q: str, prefixes: list[str]) -> None:
    assert ms.expand_query(q).icd_prefixes == prefixes


def test_icd_code_query_has_no_text_stems_or_doctor_chip() -> None:
    plan = ms.expand_query("K29.7")
    assert plan.phrase_stems == []
    parts = ms.sql_parts(plan)
    assert set(parts) == {ms.CHIP_ICD}
    clause, values = parts[ms.CHIP_ICD]
    assert "diagnosis_code GLOB" in clause and values == ["K29.7*"]
    phrase_values = ms.sql_parts(ms.expand_query("гипертония"))[ms.CHIP_PHRASE][1]
    assert phrase_values == ['("гипертон"*)'], "текст и название МКБ - одним FTS MATCH по началу слова"


def test_aliases_dictionary_is_large_and_consistent() -> None:
    raw = json.loads((ROOT / "data" / "icd_reference" / "dx_aliases_ru.json").read_text(encoding="utf-8"))
    aliases = [row["alias"] for row in raw["abbreviations"]]
    assert len(aliases) == len(set(aliases)), "дубли алиасов"
    for alias in aliases:
        assert alias == alias.strip().lower().replace("ё", "е"), alias
    assert len(raw["synonyms"]) >= 200
    assert ms.alias_pairs_count() >= 300
    ref = json.loads((ROOT / "data" / "icd_reference" / "icd10_ru_mkb10su.json").read_text(encoding="utf-8"))
    valid = {row["code"].upper() for row in ref}
    seeds = {row["seed_code"] for row in raw["abbreviations"] if row.get("seed_code")}
    seeds |= {code for group in raw["synonyms"] for code in group.get("seed_codes", [])}
    assert not (seeds - valid), sorted(seeds - valid)


@pytest.mark.parametrize(
    ("q", "expected_synonym"),
    [
        ("гипертония", "гипертензия"),
        ("АГ", "артериальная гипертензия"),
        ("близорукость", "миопия"),
        ("дорсопатия", "остеохондроз"),
        ("гэрб", "гастроэзофагеальная рефлюксная болезнь"),
        ("ОРВИ", "острая респираторная вирусная инфекция"),
        ("хобл", "хроническая обструктивная болезнь легких"),
    ],
)
def test_alias_expansion(q: str, expected_synonym: str) -> None:
    plan = ms.expand_query(q)
    assert expected_synonym in plan.synonyms, plan.synonyms


def test_term_codes_from_icd_titles_are_specific() -> None:
    hyper = ms.expand_query("гипертония").term_codes
    assert "I10" in hyper and "I11" in hyper
    assert not any(code.startswith("G9") for code in hyper), "внутричерепная гипертензия не должна тянуться из однословного синонима"
    myopia = ms.expand_query("близорукость").term_codes
    assert myopia == ["H52.1"], "миопия не должна цеплять миопатию (G71-G72)"
    assert not any("-" in code for code in ms.expand_query("острая респираторная").term_codes)


def test_fuzzy_only_when_no_exact_hits() -> None:
    typo = ms.expand_query("гипертенизя")
    assert [f["suggestion"] for f in typo.fuzzy][:1] and typo.fuzzy[0]["suggestion"].startswith("гипертенз")
    assert typo.fuzzy_stems == [["гипертенз"]]
    assert ms.expand_query("гипертония").fuzzy == [], "у слова с синонимами похожие не ищутся"
    assert ms.expand_query("остеохандроз").fuzzy[0]["suggestion"] == "остеохондроз"
    assert ms.expand_query("абв").fuzzy == []


def test_disabled_chips_are_excluded_from_sql_and_plan() -> None:
    plan = ms.expand_query("гипертония", disabled="synonyms,fuzzy")
    parts = ms.sql_parts(plan)
    assert ms.CHIP_SYNONYMS not in parts and ms.CHIP_PHRASE in parts and ms.CHIP_TERMS in parts
    payload = plan.to_dict()
    chips = {chip["id"]: chip for chip in payload["chips"] if chip}
    assert chips["synonyms"]["enabled"] is False and chips["phrase"]["enabled"] is True
    assert payload["disabled"] == ["synonyms", "fuzzy"]


def test_sql_rank_inline_has_no_placeholders_and_keeps_literals_quoted() -> None:
    import re

    plan = ms.expand_query("гипертония")
    rank = ms.sql_rank_inline(plan)
    assert "?" not in rank and rank.startswith("CASE WHEN") and "THEN 1" not in rank
    assert "THEN 2" in rank and "THEN 3" in rank and "THEN 4" in rank
    evil = ms.sql_rank_inline(ms.expand_query("x'); DROP TABLE fact_mo_case; --"))
    # Вне строковых литералов нет ни кавычек, ни точек с запятой: всё пользовательское - внутри '...'.
    outside = re.sub(r"'[^']*'", "", evil)
    assert "'" not in outside and ";" not in outside and "--" not in outside


def _mini_warehouse(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE fact_mo_case (mis_id TEXT, visit_id TEXT, visit_date TEXT, diagnosis_code TEXT,
                diagnosis_text TEXT, doctor_key TEXT);
            CREATE TABLE dim_doctor (doctor_key TEXT, doctor_fio TEXT);
            CREATE TABLE dim_diagnosis (diagnosis_code TEXT, diagnosis_label TEXT);
            INSERT INTO fact_mo_case VALUES
              ('1','1','2026-09-01','I10','Эссенциальная гипертензия','d1'),
              ('2','2','2026-09-01','I11.9','Гипертоническая болезнь 2 ст','d1'),
              ('3','3','2026-09-01','K29.7','Хронический гастрит','d2'),
              ('4','4','2026-09-01','H52.1','Миопия слабой степени','d2'),
              ('5','5','2026-09-01','M42.1','Остеохондроз поясничного отдела','d1'),
              ('6','6','2026-09-01','J06.9','ОРВИ','d2'),
              ('7','7','2026-09-01','G72.9','Миопатия неуточненная','d1'),
              ('8','8','2026-09-01','I15.0','Реноваскулярная гипертензия (АГ) 2 ст.','d2'),
              ('9','9','2026-09-01','K86.1','Хр. панкреатит','d2');
            INSERT INTO dim_doctor VALUES ('d1','Иванова А.А.'),('d2','Петров Б.Б.');
            INSERT INTO dim_diagnosis VALUES ('I10','Эссенциальная гипертензия'),('H52.1','Миопия');
            """
        )
        status = ms.ensure_search_index(conn)
        assert status == {"rows": 9, "rebuilt": True}


def _run(path: Path, q: str, disabled: str = "") -> list[tuple[str, int]]:
    plan = ms.expand_query(q, disabled=disabled)
    clause, values = ms.sql_clause(plan)
    rank = ms.sql_rank_inline(plan)
    sql = f"""
        SELECT c.mis_id, {rank} AS r FROM fact_mo_case c
        LEFT JOIN dim_doctor d ON d.doctor_key = c.doctor_key
        LEFT JOIN dim_diagnosis dx ON dx.diagnosis_code = c.diagnosis_code
        WHERE {clause} ORDER BY r, c.mis_id
    """
    with sqlite3.connect(path) as conn:
        return [(row[0], row[1]) for row in conn.execute(sql, values)]


def test_sql_clause_finds_by_code_synonym_and_typo(tmp_path: Path) -> None:
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    # Синоним: «гипертония» находит и «гипертензию» (I10) и «гипертоническую болезнь».
    hits = _run(db, "гипертония")
    ids = [m for m, _r in hits]
    assert set(ids) == {"1", "2", "8"}
    ranks = dict(hits)
    assert ranks["2"] == 2, "прямое слово запроса - ранг фразы"
    assert ranks["1"] in {3, 4}, "гипертензия - синоним или код по названию"
    # Без синонимов остаётся только фраза (и код I10 по названию).
    assert {m for m, _r in _run(db, "гипертония", disabled="synonyms,terms")} == {"2"}
    # Опечатка.
    # «2» (I11.9) - через название из справочника «Гипертензивная […] болезнь», подтянутое в label_text.
    assert {m for m, _r in _run(db, "гипертенизя")} == {"1", "2", "8"}
    # Код и диапазон.
    assert {m for m, _r in _run(db, "I10-I15")} == {"1", "2", "8"}
    assert {m for m, _r in _run(db, "K29")} == {"3"}
    # Синоним близорукость -> миопия.
    assert {m for m, _r in _run(db, "близорукость")} == {"4"}
    # Врач.
    assert {m for m, _r in _run(db, "Петров")} == {"3", "4", "6", "8", "9"}
    # Мусор ничего не ломает.
    assert _run(db, "x'); DROP TABLE fact_mo_case; --") == []


def test_match_record_mirrors_sql(tmp_path: Path) -> None:
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT c.*, d.doctor_fio, dx.diagnosis_label FROM fact_mo_case c "
                "LEFT JOIN dim_doctor d ON d.doctor_key=c.doctor_key "
                "LEFT JOIN dim_diagnosis dx ON dx.diagnosis_code=c.diagnosis_code"
            )
        ]
    for q in ("гипертония", "гипертенизя", "I10-I15", "K29", "близорукость", "Петров", "остеохондроз", "орви"):
        plan = ms.expand_query(q)
        sql_ids = {m for m, _r in _run(db, q)}
        py_ids = {r["mis_id"] for r in rows if ms.match_record(plan, r) is not None}
        assert py_ids == sql_ids, q


def test_text_stem_matches_word_start_not_substring(tmp_path: Path) -> None:
    """«близорукость» -> синоним «миопия»: стем `миопи` не должен ловить «миопатию» (Bugbot)."""
    assert ms.text_stem("миопия") == "миопи"
    assert ms.text_stem("острая") == "остр", "у прилагательных короткий стем остаётся"
    assert ms.text_stem("гипертония") == "гипертон"
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    for q in ("близорукость", "миопия"):
        ids = {m for m, _r in _run(db, q)}
        assert "4" in ids and "7" not in ids, (q, ids)
    # Совпадение только с начала слова: «тония» внутри «гипертония» не находится.
    assert {m for m, _r in _run(db, "тония")} == set()
    # Скобки и точки - границы слов и для сокращения целым словом.
    assert "8" in {m for m, _r in _run(db, "АГ")}


def test_word_expansions_apply_to_query(tmp_path: Path) -> None:
    plan = ms.expand_query("хр. панкреатит")
    assert plan.normalized == "хронический панкреатит"
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    ids = {m for m, _r in _run(db, "хронический панкреатит")}
    assert "9" in ids, "«Хр. панкреатит» в тексте должен находиться по расширенному запросу"
    assert "9" in {m for m, _r in _run(db, "хр. панкреатит")}


def test_fuzzy_is_per_token_even_when_other_word_gives_codes() -> None:
    plan = ms.expand_query("орви ринитт")
    assert plan.term_codes, "«орви» даёт коды по названию через seed-коды алиаса"
    assert plan.fuzzy and plan.fuzzy[0]["suggestion"] == "ринит", plan.fuzzy
    plan = ms.expand_query("острая респираторнная")
    assert plan.fuzzy and plan.fuzzy[0]["suggestion"].startswith("респираторн"), plan.fuzzy


def test_search_index_follows_writes_and_rebuilds_when_inconsistent(tmp_path: Path) -> None:
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    with sqlite3.connect(db) as conn:
        assert ms.ensure_search_index(conn) == {"rows": 9, "rebuilt": False}, "согласованный индекс не пересобирается"
        # Триггеры: вставка, обновление текста, удаление.
        conn.execute("INSERT INTO fact_mo_case VALUES ('10','10','2026-09-02','H52.1','Близорукость обоих глаз','d1')")
        conn.execute("UPDATE fact_mo_case SET diagnosis_text='Ёжик: гипертонический криз' WHERE mis_id='3'")
        conn.execute("DELETE FROM fact_mo_case WHERE mis_id='6'")
        conn.commit()
    assert "10" in {m for m, _r in _run(db, "близорукость")}
    assert "3" in {m for m, _r in _run(db, "ежик")}, "ё в тексте = е в запросе"
    assert "3" in {m for m, _r in _run(db, "гипертония")}
    assert {m for m, _r in _run(db, "орви")} == set()
    # Рассинхрон (например, после VACUUM или записи без триггеров) - пересборка.
    with sqlite3.connect(db) as conn:
        conn.execute(f"DELETE FROM {ms.FTS_TABLE} WHERE rowid IN (SELECT rowid FROM fact_mo_case WHERE mis_id='1')")
        conn.commit()
        assert ms.ensure_search_index(conn)["rebuilt"] is True
    assert "1" in {m for m, _r in _run(db, "гипертензия")}


def test_fts_query_only_accepts_plain_tokens() -> None:
    assert ms.fts_query(["гипертон", "i10"], ["аг"]) == '"гипертон"* AND "i10"* AND "аг"'
    assert ms.fts_query(['x" OR 1=1', "ok"]) == '"ok"*', "токены с кавычками и пробелами отбрасываются"
    inline = ms.sql_rank_inline(ms.expand_query("гипертония"))
    assert "MATCH '" in inline and "'(\"гипертон\"*)'" in inline


def test_subphrase_of_multiword_synonym_expands_group() -> None:
    """«острая респираторная» входит в термин группы ОРВИ - группа подключается целиком;
    однословная «острая» - нет (входила бы в сотни терминов)."""
    plan = ms.expand_query("острая респираторная")
    assert "орви" in plan.synonyms and any(exp.startswith("острая респираторная вирусная") for exp in plan.synonyms)
    assert ms.expand_query("острая").synonyms == []


def test_doctor_chip_searches_ids_only_for_digit_queries() -> None:
    word = ms.sql_parts(ms.expand_query("гипертония"))[ms.CHIP_DOCTOR][0]
    assert "visit_id" not in word and "dim_doctor" in word, "для слов - только ФИО через dim_doctor"
    digits = ms.sql_parts(ms.expand_query("иванова 37"))[ms.CHIP_DOCTOR][0]
    assert "CAST(c.visit_id AS TEXT) LIKE ?" in digits


def test_synonyms_use_single_match_without_dim_subquery() -> None:
    clause, values = ms.sql_parts(ms.expand_query("гипертония"))[ms.CHIP_SYNONYMS]
    assert clause.count("MATCH ?") == 1 and "dim_diagnosis" not in clause, "название МКБ - в индексе, не в подзапросе"
    assert " OR (" in values[0], "все синонимы - в одном MATCH через OR"
    assert "diagnosis_label" not in ms.sql_clause(ms.expand_query("острая респираторная", disabled="doctor"))[0]


def test_code_chips_resolve_prefixes_through_dim_diagnosis(tmp_path: Path) -> None:
    clause, values = ms.sql_parts(ms.expand_query("I10-I15"))[ms.CHIP_ICD]
    assert clause.startswith("c.diagnosis_code IN (SELECT diagnosis_code FROM dim_diagnosis WHERE")
    assert values == ["I10*", "I11*", "I12*", "I13*", "I14*", "I15*"]
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    with sqlite3.connect(db) as conn:
        # Код есть в складе, но не в справочнике: ensure_search_index добавляет его без названия.
        conn.execute("INSERT INTO fact_mo_case VALUES ('11','11','2026-09-02','I13.2','ГБ с поражением сердца и почек','d1')")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM dim_diagnosis WHERE diagnosis_code='I13.2'").fetchone()[0] == 0
        ms.ensure_search_index(conn)
        label = conn.execute("SELECT diagnosis_label FROM dim_diagnosis WHERE diagnosis_code='I13.2'").fetchone()[0]
        assert label == ms.icd_title_map()["I13.2"], "код добавлен и сразу подписан названием из справочника"
    assert ("11", 1) in _run(db, "I10-I15")


def test_missing_dim_labels_are_filled_from_icd_reference_and_searchable(tmp_path: Path) -> None:
    """Прод: mo_daily пишет коды в dim_diagnosis с пустым названием (отпечаток `2441:0`);
    ensure_search_index подтягивает названия из справочника, и они попадают в label_text."""
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    with sqlite3.connect(db) as conn:
        label = conn.execute("SELECT diagnosis_label FROM dim_diagnosis WHERE diagnosis_code='I11.9'").fetchone()[0]
        assert label.startswith("Гипертензивная"), label
        assert conn.execute("SELECT COUNT(*) FROM dim_diagnosis WHERE TRIM(COALESCE(diagnosis_label,''))=''").fetchone()[0] == 0
        assert ms.fill_missing_labels(conn) == 0, "повторный вызов ничего не трогает"
        # Уже заполненные названия не перетираются.
        assert conn.execute("SELECT diagnosis_label FROM dim_diagnosis WHERE diagnosis_code='I10'").fetchone()[0] == "Эссенциальная гипертензия"
    # Текст «Гипертоническая болезнь 2 ст» + название «Гипертензивная [гипертоническая] болезнь…»: фраза по названию.
    assert ("2", 2) in _run(db, "гипертензивная болезнь")
    assert ms.icd_title_map()["I10"] == "Эссенциальная [первичная] гипертензия"


def test_codes_for_phrase_uses_inverted_index_and_matches_full_scan() -> None:
    """Обратный индекс даёт тот же результат, что полный проход по названиям."""
    for phrase in ("гипертония", "хронический гастрит", "острая респираторная вирусная инфекция", "миопия", "нет такого"):
        stems = [ms.stem(t) for t in ms.tokens(phrase) if len(t) >= 3 and t not in ms._STOP]
        stems = [s for s in stems if len(s) >= 3]
        expected = sorted(
            code
            for code, _title, title_stems in ms._icd_titles()
            if "-" not in code and stems and all(any(ms._stem_hit(ts, s) for ts in title_stems) for s in stems)
        )
        assert sorted(ms.codes_for_phrase(phrase)) == expected, phrase
    assert ms.codes_for_phrase("миопия") and not any(c.startswith("G72") for c in ms.codes_for_phrase("миопия"))


def test_search_index_rebuilds_on_schema_version_and_dim_change(tmp_path: Path) -> None:
    db = tmp_path / "w.sqlite"
    _mini_warehouse(db)
    with sqlite3.connect(db) as conn:
        # Название МКБ попало в индекс: «миопия» из dim_diagnosis находит строку с текстом «Близорукость».
        conn.execute("INSERT INTO fact_mo_case VALUES ('10','10','2026-09-02','H52.1','Близорукость','d1')")
        conn.commit()
    assert ("10", 2) in _run(db, "миопия"), "ранг 2 (фраза) возможен только через название МКБ в индексе"
    with sqlite3.connect(db) as conn:
        # Словарь диагнозов поменялся -> отпечаток другой -> пересборка подтягивает новые названия.
        conn.execute("UPDATE dim_diagnosis SET diagnosis_label='Хронический панкреатит' WHERE diagnosis_code='K86.1'")
        assert conn.total_changes >= 1, "код K86.1 уже добавлен страховкой инварианта без названия"
        conn.commit()
        assert ms.ensure_search_index(conn)["rebuilt"] is True
        assert ms.ensure_search_index(conn)["rebuilt"] is False
    assert ("9", 2) in _run(db, "хронический панкреатит"), "«хронический» - из названия МКБ, «панкреатит» - из текста"
    with sqlite3.connect(db) as conn:
        # Старая версия схемы - индекс и триггеры сносятся и создаются заново.
        conn.execute(f"UPDATE {ms.FTS_META_TABLE} SET v='1' WHERE k='schema_version'")
        conn.commit()
        assert ms.ensure_search_index(conn)["rebuilt"] is True
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({ms.FTS_TABLE})")]
        assert cols == ["search_text", "label_text", "mis_id"]
        trg = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert set(ms._FTS_TRIGGERS) <= trg
    assert "10" in {m for m, _r in _run(db, "миопия")}


def test_search_index_upgrades_from_v1_layout(tmp_path: Path) -> None:
    """Прод после релиза 5: таблица без label_text и без meta - ensure_search_index её заменяет."""
    db = tmp_path / "w.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE fact_mo_case (mis_id TEXT, visit_id TEXT, visit_date TEXT, diagnosis_code TEXT,
                diagnosis_text TEXT, doctor_key TEXT);
            CREATE TABLE dim_doctor (doctor_key TEXT, doctor_fio TEXT);
            CREATE TABLE dim_diagnosis (diagnosis_code TEXT, diagnosis_label TEXT);
            INSERT INTO fact_mo_case VALUES ('1','1','2026-09-01','I10','Эссенциальная гипертензия','d1');
            CREATE VIRTUAL TABLE fact_mo_case_search USING fts5(search_text, mis_id UNINDEXED);
            INSERT INTO fact_mo_case_search(rowid, search_text, mis_id) VALUES (1, 'Эссенциальная гипертензия', '1');
            CREATE TRIGGER trg_fact_mo_case_search_ad AFTER DELETE ON fact_mo_case BEGIN
              DELETE FROM fact_mo_case_search WHERE rowid = old.rowid; END;
            """
        )
        assert ms.ensure_search_index(conn) == {"rows": 1, "rebuilt": True}
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({ms.FTS_TABLE})")]
        assert cols == ["search_text", "label_text", "mis_id"]
    assert "1" in {m for m, _r in _run(db, "гипертензия")}


def test_warm_caches_loads_dictionaries() -> None:
    stats = ms.warm_caches()
    assert stats["icd_titles"] > 10000 and stats["stems"] > 5000 and stats["aliases"] > 100 and stats["vocab"] > 1000


def test_fuzzy_chip_skipped_when_all_stem_lists_empty() -> None:
    plan = ms.expand_query("гипертенизя")
    plan.fuzzy_stems = [[]]
    assert ms.CHIP_FUZZY not in ms.sql_parts(plan)


def test_suggest_returns_aliases_words_and_typos() -> None:
    items = ms.suggest("гиперт")
    labels = [item["label"] for item in items]
    assert labels and len(items) <= 8
    assert any(label.startswith("гиперт") for label in labels)
    assert ms.suggest("г") == []
    typo = ms.suggest("остеохандр")
    assert any(item["kind"] == "typo" and item["label"] == "остеохондроз" for item in typo)


def test_search_plan_payload_shape() -> None:
    payload = ms.expand_query("гипертония").to_dict()
    chips = [chip for chip in payload["chips"] if chip]
    ids = [chip["id"] for chip in chips]
    assert ids == ["phrase", "synonyms", "terms", "doctor"]
    for chip in chips:
        assert chip["label"] and isinstance(chip["values"], list) and chip["enabled"] is True
