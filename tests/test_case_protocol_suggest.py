from __future__ import annotations

from clinical_knowledge.case_protocol_suggest import (
    build_case_fact_graph,
    suggest_protocols_for_case,
)
from clinical_knowledge.protocol_links import protocol_display_name, title_looks_truncated


def test_build_case_fact_graph_age_from_bdate_and_visit() -> None:
    graph = build_case_fact_graph(
        clinical={"clinical_diagnosis": "плоскостопие", "patient_bdate": "2019-03-01"},
        record={"visit_id": "v-age", "visit_date": "2026-07-15", "specialty": "Ортопед"},
    )
    assert graph["audience"] == "child"
    assert graph["age_years"] == 7
    assert graph["visit_date"] == "2026-07-15"
    assert graph["age_source"] == "bdate_visit"


def test_suggest_adult_does_not_pick_det_nas() -> None:
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Геморрой неуточненный",
            "mis_diagnos": "I84.9",
            "patient_age_years": 41,
            "visit_date": "2026-07-15",
        },
        record={"visit_id": "adult-kp", "date": "2026-07-15"},
        limit=3,
    )
    paths = " ".join(str(item.get("source_path") or "") for item in result.get("items") or []).lower()
    assert "дет_нас" not in paths
    assert "детс_нас" not in paths


def test_build_case_fact_graph_uses_diagnosis_text_not_icd() -> None:
    graph = build_case_fact_graph(
        clinical={
            "complaints": "боль в колене; хромота",
            "clinical_diagnosis": "M60 Миозит прямой мышцы бедра",
            "exam_recommendations": "УЗИ",
            "patient_age_years": 7,
        },
        record={"visit_id": "v1", "specialty": "Ортопед"},
        findings=[
            {"code": "B_dx_no_support", "title_ru": "Нет обоснования диагноза"},
            {"code": "D_reg55_p0", "title_ru": "Критический дефект по №55"},
            {"code": "C_nsaid_dup", "title_ru": "Одновременно ≥2 НПВП"},
            {"code": "B_icd_dir_no_match", "title_ru": "не в справочнике"},
        ],
    )
    assert graph["case_id"] == "v1"
    assert graph["diagnoses"]
    assert "Миозит" in graph["diagnoses"][0]["text"]
    assert "M60" in (graph["diagnoses"][0].get("icd10") or [])
    assert "боль в колене" in graph["complaints"]
    assert graph["audience"] == "child"
    assert any(g.get("code") == "B_dx_no_support" for g in graph["gaps"])
    assert not any(str(g.get("code") or "").startswith("D_reg55") for g in graph["gaps"])
    assert not any(str(g.get("code") or "").startswith("C_nsaid") for g in graph["gaps"])
    assert not any(str(g.get("code") or "").startswith("B_icd") for g in graph["gaps"])
    assert graph["specialty"]["slug"] == "travmatologiya-ortopediya"


def test_suggest_flatfoot_child_hits_pediatric_ortho_kp() -> None:
    """Реестр: детский ортопедо-травматологический КП (пост. 109) по тексту Dx."""
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": (
                "Плосковальгусная установка стоп без нарушения опоры и передвижения "
                "с вальгированием пяточных костей."
            ),
            "mis_diagnos": "M21.0",
            "diagnosis_main_text": (
                "M21.0. Вальгусная деформация, не классифицированная в других рубриках"
            ),
            "patient_age_years": 7,
        },
        record={"visit_id": "3605554", "specialty": "Ортопед-травматолог"},
        limit=3,
    )
    assert result["available"] is True
    assert result["items"]
    top = result["items"][0]
    path = (top.get("source_path") or "").lower()
    assert top["match_kind"] == "clinical"
    assert "детс" in path
    assert "ортопедо" in path or "травматолог" in path
    assert all(item["match_kind"] == "clinical" for item in result["items"])


def test_truncated_registry_title_uses_filename() -> None:
    assert title_looks_truncated('клинический протокол «Диагностика и лечение')
    title = protocol_display_name(
        "minzdrav_protocols/urologiya/KP_cystitis_adult.pdf",
        registry_title='клинический протокол «Диагностика и лечение',
        prefer_filename_if_truncated=True,
    )
    assert "cystitis" in title.lower() or "KP" in title or "цист" in title.lower() or len(title) > 10
    assert "Диагностика и лечение" not in title or "»" in title


def test_suggest_protocols_returns_contract(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")
    captured: dict = {}

    def _fake_match(facts, specialty_slug=None, limit=8, use_icd=False):
        captured["facts"] = facts
        captured["use_icd"] = use_icd
        return [
            {
                "protocol_id": "p1",
                "title": "Миозит и заболевания мышц",
                "source_path": "minzdrav_protocols/travmatologiya-ortopediya/KP_myositis.pdf",
                "match_score": 77.0,
                "icd_fit": [],
                "icd_fit_label": "",
                "specialty_slug": "travmatologiya-ortopediya",
            }
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards",
        _fake_match,
    )
    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards_by_diagnosis_text",
        lambda facts, specialty_slug=None, limit=8: _fake_match(
            facts, specialty_slug=specialty_slug, limit=limit, use_icd=False
        ),
    )
    result = suggest_protocols_for_case(
        clinical={"clinical_diagnosis": "миозит прямой мышцы", "complaints": "боль"},
        record={"visit_id": "v2", "specialty": "Ортопед"},
        findings=[{"code": "D_reg55_p0", "title_ru": "Критический дефект по №55"}],
        limit=3,
    )
    assert result["ok"] is True
    assert result["available"] is True
    assert result["engine"] == "case_protocol_suggest_v5"
    assert result["mode"] == "text"
    assert captured["facts"]["consultation"]["icd10"] == []
    item = result["items"][0]
    assert item["protocol_id"] == "p1"
    assert item["match_kind"] == "clinical"
    assert item["match_kind_label"]
    assert any(r.get("code") == "diagnosis_fit" for r in item["reasons"])
    assert not any("МКБ" in (r.get("text") or "") for r in item["reasons"])
    assert not any("Критический дефект" in (r.get("text") or "") for r in item["reasons"])
    assert item["viewer_url"].startswith("/proto-viewer.html?path=")
    assert item["search_url"].startswith("/doctor/search?q=")


def test_suggest_prefers_diagnosis_text_over_wrong_icd_seed(monkeypatch) -> None:
    """Чужой код + противоречащий текст → text-path, не ICD-first по M60."""
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")
    monkeypatch.setattr(
        "icd_mkb.is_code_in_ru_reference",
        lambda code: str(code).upper().startswith("M60"),
    )
    monkeypatch.setattr("icd_mkb.ru_title", lambda code: "Миозит")

    def _fake_match(facts, specialty_slug=None, limit=8):
        assert facts["consultation"]["icd10"] == []
        dx = (facts["consultation"].get("diagnosis_text") or "").lower()
        assert "ювенильн" in dx or "артрит" in dx
        return [
            {
                "protocol_id": "jra",
                "title": "Ювенильный артрит иммуновоспалительные заболевания",
                "source_path": "minzdrav_protocols/revmatologiya/KP_jra_ped.pdf",
                "match_score": 82.0,
                "icd_fit": [],
                "specialty_slug": specialty_slug or "revmatologiya",
            },
            {
                "protocol_id": "soft",
                "title": "Инфекции мягких тканей M60",
                "source_path": "minzdrav_protocols/khirurgiya/KP_soft_tissue.pdf",
                "match_score": 40.0,
                "icd_fit": [],
                "specialty_slug": "khirurgiya",
            },
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards_by_diagnosis_text",
        _fake_match,
    )
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Ювенильный идиопатический артрит, код M60 ошибочно",
            "complaints": "хромота 3 месяца",
        },
        record={"visit_id": "smirnova", "specialty": "Педиатр"},
        limit=2,
    )
    assert result["mode"] == "text"
    assert result["items"][0]["protocol_id"] == "jra"
    assert not any(r.get("code") == "icd_fit" for r in result["items"][0]["reasons"])


def test_suggest_blocks_stomatology_for_urologist(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")

    def _fake_match(facts, specialty_slug=None, limit=8):
        return [
            {
                "protocol_id": "bad",
                "title": "Заболевания челюстно-лицевой области",
                "source_path": "minzdrav_protocols/stomatologiya/chelust.pdf",
                "match_score": 90.0,
                "icd_fit": [],
                "specialty_slug": "stomatologiya",
            },
            {
                "protocol_id": "good",
                "title": "Урология взрослых циркумцизио",
                "source_path": "minzdrav_protocols/urologiya/kp_urology.pdf",
                "match_score": 70.0,
                "icd_fit": [],
                "specialty_slug": "urologiya",
            },
        ]

    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards_by_diagnosis_text",
        _fake_match,
    )
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Состояние после циркумцизио",
            "complaints": "боль в ране",
            "treatment_recommendations": 'Ибупрофен ("Кетопрофен" и т.п.)',
            "doctor_specialization": "Уролог",
        },
        record={"visit_id": "3650612"},
        findings=[{"code": "C_nsaid_dup", "title_ru": "Одновременно ≥2 НПВП"}],
        limit=3,
    )
    ids = [item.get("protocol_id") for item in result.get("items") or []]
    assert "bad" not in ids
    assert "good" in ids


def test_suggest_f41_2_psychotherapist_icd_first() -> None:
    """F41.2 code-only + психотерапевт → КП невротические/стресс, не вены."""
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "F41.2",
            "mis_diagnos": "F41.2",
            "patient_age_years": 34,
        },
        record={"visit_id": "f41-psych", "specialty": "Психотерапевт"},
        limit=3,
    )
    assert result["available"] is True
    assert result["mode"] == "icd_first"
    assert result["engine"] == "case_protocol_suggest_v5"
    top = result["items"][0]
    path = (top.get("source_path") or "").lower()
    title = (top.get("title") or "").lower()
    assert top["match_kind"] == "clinical"
    assert top["score"] >= 40
    assert "psikhiatriya" in path
    assert "неврот" in path or "стресс" in path or "неврот" in title or "стресс" in title
    assert "вен" not in path and "тромбофил" not in path and "многоплод" not in path
    blob = " ".join(
        str(r.get("text") or "") for r in (top.get("reasons") or [])
    ).lower()
    assert "мкб" in blob or any(r.get("code") == "icd_fit" for r in top.get("reasons") or [])


def test_resolve_kp_query_cascade() -> None:
    from clinical_knowledge.case_protocol_suggest import resolve_kp_query

    by_dx = resolve_kp_query(diag_text="Геморрой неуточненный", codes_in_dir=["K64.9"])
    assert by_dx["source"] == "diagnosis"
    assert by_dx["use_icd"] is False
    by_icd = resolve_kp_query(diag_text="", codes_in_dir=["F41.2"])
    assert by_icd["source"] == "icd"
    assert by_icd["use_icd"] is True
    by_complaints = resolve_kp_query(
        diag_text="",
        codes_in_dir=[],
        graph={
            "complaints": ["боль в заднем проходе", "кровь при дефекации"],
            "anamnesis": "симптомы несколько месяцев",
        },
    )
    assert by_complaints["source"] == "none"
    empty = resolve_kp_query(diag_text="", codes_in_dir=[], graph={})
    assert empty["source"] == "none"


def test_suggest_i84_9_does_not_fill_with_aneurysm() -> None:
    """I84.9 (геморрой) → КП прямой кишки по диагнозу/содержанию, не аорта/ОПН."""
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Геморрой неуточненный",
            "mis_diagnos": "I84.9",
            "patient_age_years": 41,
        },
        record={"visit_id": "3676800"},
        limit=3,
    )
    titles = " ".join(str(item.get("title") or "") for item in result.get("items") or []).lower()
    paths = " ".join(str(item.get("source_path") or "") for item in result.get("items") or []).lower()
    blob = titles + " " + paths
    assert "аневризм" not in blob
    assert "почечн" not in blob
    assert all(item.get("match_kind") == "clinical" for item in result.get("items") or [])
    assert result.get("query_source") == "diagnosis"
    assert result.get("available") is True
    assert result.get("items")
    assert "прямой" in blob or "параректал" in blob or "22" in blob


def test_suggest_empty_says_no_protocol(monkeypatch) -> None:
    monkeypatch.setenv("CASE_PROTOCOL_SUGGEST", "1")
    monkeypatch.setattr(
        "clinical_knowledge.protocol_match.match_protocol_cards_by_diagnosis_text",
        lambda facts, specialty_slug=None, limit=8: [],
    )
    result = suggest_protocols_for_case(
        clinical={"clinical_diagnosis": "Редкая нозология xyzzy без протокола"},
        record={"visit_id": "no-kp"},
        limit=3,
    )
    assert result["available"] is False
    assert result["items"] == []
    assert "Нет клинического протокола" in (result.get("reason") or "")


def test_protocol_dedup_collapses_same_pdf_in_two_rubrics() -> None:
    from clinical_knowledge.case_protocol_suggest import _dedup_protocol_rows

    rows = [
        {
            "protocol_id": "a",
            "title": "Аневризма",
            "source_path": "minzdrav_protocols/khirurgiya/КП аневризма 01.06.2017 № 47.pdf",
            "approval": {"date": "2017-06-01", "number": "47"},
        },
        {
            "protocol_id": "b",
            "title": "Аневризма",
            "source_path": (
                "minzdrav_protocols/bolezni-sistemy-krovoobrashcheniya/"
                "КП аневризма 01.06.2017 № 47.pdf"
            ),
            "approval": {"date": "2017-06-01", "number": "47"},
        },
    ]
    out = _dedup_protocol_rows(rows)
    assert len(out) == 1


def test_i84_is_not_venous_leg_boost() -> None:
    from clinical_knowledge.protocol_match import _is_venous_icd

    assert _is_venous_icd(["I83.9"]) is True
    assert _is_venous_icd(["I84.9"]) is False
    assert _is_venous_icd(["K64.9"]) is False


def test_suggest_without_diagnosis_is_empty() -> None:
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "complaints": "кашель сухой 3 дня",
            "anamnesis_doctor": "болеет неделю без температуры",
            "patient_age_years": 40,
        },
        record={"visit_id": "no-dx"},
        limit=3,
    )
    assert result["available"] is False
    assert not result.get("items")


def test_j06_does_not_pick_rare_or_omnibus_ent() -> None:
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "ОРВИ",
            "mis_diagnos": "J06.9",
            "patient_age_years": 34,
        },
        record={"visit_id": "j06", "date": "2026-08-02"},
        limit=3,
    )
    blob = " ".join(
        str(item.get("source_path") or "") + " " + str(item.get("title") or "")
        for item in result.get("items") or []
    ).lower()
    assert "цилиарн" not in blob
    assert "гемопоэтическ" not in blob
    assert "оториноларинголог" not in blob
    assert "2017_49" not in blob
    assert "экстренн" not in blob
    assert "злокачествен" not in blob
    assert "онколог" not in blob
    assert "2018_17" not in blob
    assert "2018_60" not in blob
    assert "нейрохирургическ" not in blob
    assert "миелом" not in blob


def test_dorsalgia_does_not_pick_neuro_omnibus() -> None:
    import os

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical={
            "clinical_diagnosis": "Дорсалгия",
            "mis_diagnos": "M54.5",
            "patient_age_years": 42,
        },
        record={"visit_id": "m54", "date": "2026-07-15", "specialty": "Невролог"},
        limit=3,
    )
    blob = " ".join(
        str(item.get("source_path") or "") + " " + str(item.get("title") or "")
        for item in result.get("items") or []
    ).lower()
    assert "нейрохирургическ" not in blob
    assert result.get("available") is False
