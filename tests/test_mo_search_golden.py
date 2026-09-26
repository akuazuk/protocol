"""Волна D: умный поиск через `build_cases` (SQL и Python-путь), `search_plan`, `search_off`,
ранжирование и золотой набор запросов из плана 2026-09-26 §D."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend, mo_search
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse

PERIOD = {"period": "custom", "date_from": "2026-07-01", "date_to": "2026-07-31"}

CASES = [
    # mis_id, code, text
    ("h1", "I10", "Эссенциальная гипертензия"),
    ("h2", "I11.9", "Гипертоническая болезнь II ст., риск 3"),
    ("h3", "I10", "АГ 2 ст."),
    ("h4", "I15.0", "Реноваскулярная гипертензия"),
    ("g1", "K29.7", "Хронический гастрит, обострение"),
    ("g2", "K21.0", "ГЭРБ с эзофагитом"),
    ("m1", "H52.1", "Миопия слабой степени OU"),
    ("m2", "H52.1", "Близорукость средней степени"),
    ("o1", "M42.1", "Остеохондроз поясничного отдела позвоночника"),
    ("o2", "M51.1", "Дорсопатия. Грыжа диска L5-S1"),
    ("r1", "J06.9", "ОРВИ, острый ринофарингит"),
    ("r2", "J06.9", "Острая респираторная вирусная инфекция"),
    ("r3", "J20.9", "Острый бронхит"),
    ("d1", "E11.9", "Сахарный диабет 2 типа"),
    ("z1", "E04.1", "Узловой зоб"),
]


def _seed(path: Path) -> None:
    initialize_warehouse(path)
    doc_a = doctor_key_for("Иванова Анна")
    doc_b = doctor_key_for("Петров Борис")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [(doc_a, "Иванова Анна", "Терапия", "Центр"), (doc_b, "Петров Борис", "Неврология", "Юг")],
        )
        conn.executemany(
            "INSERT INTO dim_diagnosis(diagnosis_code,diagnosis_label) VALUES(?,?)",
            [("I10", "Эссенциальная [первичная] гипертензия"), ("H52.1", "Миопия"), ("M42.1", "Остеохондроз позвоночника у взрослых")],
        )
        for index, (mis_id, code, text) in enumerate(CASES):
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,doctor_key,
                    specialty,filial,diagnosis_code,diagnosis_text,icd_chapter,content_hash,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    str(5000 + index),
                    f"2026-07-{1 + index:02d}",
                    "clinical_visit",
                    60.0 + index,
                    "ok",
                    doc_a if index % 2 else doc_b,
                    "Терапия" if index % 2 else "Неврология",
                    "Центр" if index % 2 else "Юг",
                    code,
                    text,
                    code[0],
                    f"h{index}",
                    "2026-07-31T00:00:00Z",
                ),
            )
        conn.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "mo.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


# case_id в ответе = visit_id; в ожиданиях удобнее короткие имена.
_VISIT_TO_NAME = {str(5000 + index): mis_id for index, (mis_id, _code, _text) in enumerate(CASES)}


def _ids(result: dict) -> list[str]:
    return [_VISIT_TO_NAME.get(str(row["case_id"]), str(row["case_id"])) for row in result["rows"]]


GOLDEN = [
    # запрос -> обязательные находки, запрещённые находки
    ("гипертония", {"h1", "h2", "h4"}, {"g1", "m1", "o1"}),
    ("гипертензия", {"h1", "h2", "h4"}, {"g1"}),
    ("АГ", {"h1", "h2", "h3", "h4"}, {"g1", "m1"}),
    ("гипертенизя", {"h1", "h4"}, {"g1", "m1"}),
    ("I10", {"h1", "h3"}, {"h2", "h4", "g1"}),
    ("I10-I15", {"h1", "h2", "h3", "h4"}, {"g1", "m1"}),
    ("i1", {"h1", "h2", "h3", "h4"}, {"g1"}),
    ("близорукость", {"m1", "m2"}, {"h1", "o1"}),
    ("миопия", {"m1", "m2"}, {"h1"}),
    ("дорсопатия", {"o1", "o2"}, {"h1", "m1"}),
    ("остеохондроз", {"o1", "o2"}, {"h1"}),
    ("остеохандроз", {"o1"}, {"h1", "m1"}),
    ("орви", {"r1", "r2"}, {"h1", "r3"}),
    ("острая респираторная", {"r2"}, {"h1", "g1"}),
    ("гэрб", {"g2"}, {"h1"}),
    ("рефлюкс", {"g2"}, {"h1", "g1"}),
    ("гастрит", {"g1"}, {"g2", "h1"}),
    ("K29", {"g1"}, {"g2"}),
    ("сд 2", {"d1"}, {"h1"}),
    ("диабет", {"d1"}, {"h1"}),
    ("узловой зоб", {"z1"}, {"h1"}),
    ("Петров", {"h1", "h3", "g1"}, {"h2"}),
]


@pytest.mark.parametrize(("q", "must", "must_not"), GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_queries_sql_path(warehouse: Path, q: str, must: set[str], must_not: set[str]) -> None:
    result = mo_backend.build_cases({**PERIOD, "q": q, "page_size": "100"})
    ids = set(_ids(result))
    assert must <= ids, (q, sorted(ids))
    assert not (must_not & ids), (q, sorted(must_not & ids))
    assert result["total"] == len(ids)
    plan = result["search_plan"]
    assert plan and plan["q"] == q and plan["chips"], "search_plan обязателен при текстовом поиске"
    assert sum(chip["count"] for chip in plan["chips"]) == len(ids), "счётчики чипов складываются в total"


@pytest.mark.parametrize(("q", "must", "must_not"), GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_queries_python_path_matches_sql(warehouse: Path, monkeypatch, q: str, must: set[str], must_not: set[str]) -> None:
    sql_ids = set(_ids(mo_backend.build_cases({**PERIOD, "q": q, "page_size": "100"})))
    monkeypatch.setattr(mo_backend, "_cases_sql_pageable", lambda params: False)
    py_result = mo_backend.build_cases({**PERIOD, "q": q, "page_size": "100"})
    assert set(_ids(py_result)) == sql_ids, q
    assert py_result["search_plan"] and sum(c["count"] for c in py_result["search_plan"]["chips"]) == len(sql_ids)


def test_relevance_rank_puts_code_hits_first(warehouse: Path) -> None:
    result = mo_backend.build_cases({**PERIOD, "q": "I10", "page_size": "100"})
    assert _ids(result)[:2] == ["h3", "h1"] or set(_ids(result)[:2]) == {"h1", "h3"}
    # Текст: прямое слово раньше синонима.
    result = mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"})
    ids = _ids(result)
    assert ids[0] == "h2", ids  # «Гипертоническая болезнь» - фраза
    assert set(ids[1:]) == {"h1", "h3", "h4"}  # гипертензия - синоним, I10 - код по названию
    # Явная сортировка по баллу отключает ранжирование.
    by_score = mo_backend.build_cases({**PERIOD, "q": "гипертония", "sort_by": "overall", "page_size": "100"})
    assert _ids(by_score) == ["h4", "h3", "h2", "h1"]


def test_search_off_disables_chips(warehouse: Path) -> None:
    full = mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"})
    assert set(_ids(full)) == {"h1", "h2", "h3", "h4"}
    no_syn = mo_backend.build_cases({**PERIOD, "q": "гипертония", "search_off": "synonyms,terms", "page_size": "100"})
    assert set(_ids(no_syn)) == {"h2"}
    chips = {chip["id"]: chip for chip in no_syn["search_plan"]["chips"]}
    assert chips["synonyms"]["enabled"] is False and chips["phrase"]["enabled"] is True
    assert no_syn["search_plan"]["disabled"] == ["synonyms", "terms"]


def test_cyrillic_case_and_yo_do_not_matter(warehouse: Path) -> None:
    upper = mo_backend.build_cases({**PERIOD, "q": "ГИПЕРТОНИЯ", "page_size": "100"})
    lower = mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"})
    assert set(_ids(upper)) == set(_ids(lower))
    assert set(_ids(mo_backend.build_cases({**PERIOD, "q": "Близорукость", "page_size": "100"}))) == {"m1", "m2"}


def test_search_index_is_revalidated_after_ttl(warehouse: Path, monkeypatch) -> None:
    """Долгоживущий воркер сам лечит FTS-индекс: повреждение -> следующий поиск после TTL
    пересобирает индекс (Bugbot: одноразовый кэш пропускал перепроверку)."""
    assert set(_ids(mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"}))) == {"h1", "h2", "h3", "h4"}
    with sqlite3.connect(warehouse) as conn:
        conn.execute(f"DELETE FROM {mo_search.FTS_TABLE}")
        conn.commit()
    # Внутри TTL проверка не повторяется - индекс пуст, остаются только коды и названия МКБ.
    broken = mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"})
    assert "h4" not in _ids(broken), _ids(broken)
    monkeypatch.setattr(mo_backend, "_SEARCH_INDEX_CHECKED_AT", 0.0)
    healed = mo_backend.build_cases({**PERIOD, "q": "гипертония", "page_size": "100"})
    assert set(_ids(healed)) == {"h1", "h2", "h3", "h4"}


def test_identity_lookup_still_wins_over_text_plan(warehouse: Path) -> None:
    result = mo_backend.build_cases({**PERIOD, "q": "5000", "page_size": "100"})
    assert _ids(result) == ["h1"]
    assert result["search_plan"] is None


def test_api_cases_search_off_and_suggest_and_plan(warehouse: Path, monkeypatch) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "mo-test-token"}
    base = "/api/methodist/mo/cases?period=custom&date_from=2026-07-01&date_to=2026-07-31&page_size=100"
    payload = client.get(base + "&q=гипертония&search_off=synonyms", headers=headers).json()
    assert set(_ids(payload)) == {"h1", "h2", "h3"}, "без синонимов I10 всё ещё находится кодом по названию"
    assert payload["search_plan"]["disabled"] == ["synonyms"]
    assert client.get(base + "&q=x&search_off=DROP", headers=headers).status_code == 422
    suggest = client.get("/api/methodist/mo/search/suggest?q=гиперт", headers=headers).json()
    assert suggest["engine"] == "mo_search_v1" and suggest["items"] and len(suggest["items"]) <= 8
    assert client.get("/api/methodist/mo/search/suggest?q=гиперт").status_code == 403
    plan = client.get("/api/methodist/mo/search/plan?q=I10-I15", headers=headers).json()
    assert plan["plan"]["chips"][0]["id"] == "icd" and len(plan["plan"]["chips"][0]["values"]) == 6


def test_alias_dictionary_meets_plan_threshold() -> None:
    assert mo_search.alias_pairs_count() >= 300
