from __future__ import annotations

from clinical_knowledge.case_protocol_suggest import (
    build_case_fact_graph,
    suggest_protocols_for_case,
)
from clinical_knowledge.protocol_links import protocol_display_name, title_looks_truncated


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
