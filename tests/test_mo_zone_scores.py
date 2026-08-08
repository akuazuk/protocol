"""Движок зон МО: оформление / диагноз / план по протоколу."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.mo_daily import initialize_warehouse, upsert_warehouse
from clinical_knowledge.mo_rubric_mz import load_rubric_config
from clinical_knowledge.mo_zone_scores import compute_mo_zone_scores, warehouse_zone_columns


def _rich_clinical() -> dict:
    return {
        "complaints": (
            "Боль в горле локально справа, характер ноющий, длительность 3 дня, "
            "интенсивность умеренная"
        ),
        "anamnesis_doctor": (
            "Болеет третий день. Температура до 37.5. Аллергологический анамнез не отягощён. "
            "Курение отрицает. Наследственность не отягощена. Ранее ангины ежегодно."
        ),
        "objective_status": (
            "Состояние удовлетворительное. Кожные покровы чистые. Дыхание везикулярное. "
            "ЧСС 72. А/Д 120/80. Живот мягкий. Локальный статус: гиперемия миндалин."
        ),
        "clinical_diagnosis": "Острый тонзиллит J03.9",
        "exam_recommendations": "ОАК, мазок из зева",
        "treatment_recommendations": "Полоскание, контроль через 5 дней, явка к терапевту",
        "exam_data": "ОАК от 01.08: лейкоциты 9.1",
    }


def test_rubric_yaml_has_zones_and_requires_protocol() -> None:
    load_rubric_config.cache_clear()
    cfg = load_rubric_config()
    by_id = {c["id"]: c for c in cfg["criteria"]}
    assert by_id["complaints"]["zone"] == "documentation"
    assert by_id["complaints"]["requires_protocol"] is False
    assert by_id["diagnosis"]["zone"] == "diagnosis"
    assert by_id["exam_plan"]["requires_protocol"] is True
    assert by_id["exam_data"]["optional"] is True


def test_good_case_with_kp_ok_bands() -> None:
    suggest = {
        "ok": True,
        "items": [
            {
                "title": "Острый тонзиллит",
                "match_kind": "clinical",
                "score": 80,
                "protocol_id": "demo",
            }
        ],
    }
    zones = compute_mo_zone_scores(
        {
            "clinical": _rich_clinical(),
            "meta": {
                "visit_date": "2026-08-02",
                "visit_time": "10:30",
                "diagnosis_code": "J03.9",
            },
            "block_scores": {"exams": 70, "treatment": 65},
            "protocol_suggest": suggest,
            "findings": [],
            "document_kind": "clinical_visit",
            "score_eligible": True,
        }
    )
    assert zones["ok"] is True
    assert zones["engine"] == "mo_zones_v1"
    assert zones["zone1_band"] in {"ok", "weak"}
    assert zones["zone2a_band"] in {"ok", "weak"}
    assert zones["zone2b_kp_status"] == "matched"
    assert zones["zone2b_band"] != "na"
    assert zones["zone1"]["label_ru"] == "Оформление"
    assert zones["zone2b"]["label_ru"] == "План по протоколу"
    by_id = {c["id"]: c for c in zones["criteria"]}
    assert by_id["exam_correction"]["score"] is None
    assert "не по протоколу" not in (by_id["complaints"].get("reason") or "").lower()


def test_empty_complaints_zone1_bad_no_kp_wording() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": {
                "clinical_diagnosis": "ОРВИ",
                "exam_recommendations": "ОАК",
            },
            "meta": {"visit_date": "2026-08-02", "visit_time": "09:00"},
            "findings": [],
            "document_kind": "clinical_visit",
        }
    )
    assert zones["zone1_band"] == "bad"
    blob = json.dumps(zones["criteria"], ensure_ascii=False).lower()
    assert "не соответствует протоколу" not in blob
    assert "не по протоколу" not in blob


def test_diagnosis_empty_zone2a_bad() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": {
                "complaints": "Кашель сухой 2 дня, локально за грудиной",
                "anamnesis_doctor": "Болеет второй день без температуры. Аллергии отрицает.",
                "objective_status": "Состояние удовлетворительное. Дыхание везикулярное. ЧСС 70.",
            },
            "meta": {"visit_date": "2026-08-02", "visit_time": "11:00"},
            "document_kind": "clinical_visit",
        }
    )
    assert zones["zone2a_band"] == "bad"
    assert zones["attention_primary"] in {"zone2a", "zone1"}


def test_plan_empty_with_kp_zone2b_bad() -> None:
    suggest = {
        "items": [
            {
                "title": "Пневмония",
                "match_kind": "clinical",
                "score": 85,
                "protocol_id": "p1",
            }
        ]
    }
    zones = compute_mo_zone_scores(
        {
            "clinical": {
                **_rich_clinical(),
                "exam_recommendations": "",
                "treatment_recommendations": "",
                "clinical_diagnosis": "Пневмония J18.9",
            },
            "meta": {
                "visit_date": "2026-08-02",
                "visit_time": "12:00",
                "diagnosis_code": "J18.9",
            },
            "block_scores": {"exams": 10, "treatment": 10},
            "protocol_suggest": suggest,
            "document_kind": "clinical_visit",
        }
    )
    assert zones["zone2b_kp_status"] == "matched"
    assert zones["zone2b_band"] == "bad"


def test_plan_without_kp_not_fake_protocol_fail() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": {
                **_rich_clinical(),
                "exam_recommendations": "ОАК",
                "treatment_recommendations": "Симптоматически, явка через 5 дней",
            },
            "meta": {
                "visit_date": "2026-08-02",
                "visit_time": "10:00",
                "diagnosis_code": "J03.9",
            },
            "protocol_suggest": {"items": []},
            "document_kind": "clinical_visit",
        }
    )
    assert zones["zone2b_kp_status"] == "unmatched"
    by_id = {c["id"]: c for c in zones["criteria"]}
    assert by_id["exam_plan"]["score"] is None
    assert by_id["exam_plan"].get("na_reason") == "kp_not_matched"
    assert zones["zone2b_band"] == "na"
    assert zones["zone2b"]["band_label_ru"] == "протокол не подобран"


def test_correction_without_prior_is_na() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": _rich_clinical(),
            "meta": {"visit_date": "2026-08-02", "visit_time": "10:00", "diagnosis_code": "J03.9"},
            "document_kind": "clinical_visit",
        }
    )
    by_id = {c["id"]: c for c in zones["criteria"]}
    assert by_id["exam_correction"]["score"] is None
    assert by_id["treatment_correction"]["score"] is None


def test_non_clinical_skipped() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": _rich_clinical(),
            "document_kind": "procedure_session",
            "score_eligible": False,
        }
    )
    assert zones.get("skipped") is True
    assert zones["zone1_pct"] is None
    assert zones["attention_primary"] == "none"


def test_warehouse_persists_zone_columns(tmp_path: Path) -> None:
    path = tmp_path / "wh.sqlite"
    initialize_warehouse(path)
    raw_rows = [
        {
            "id": "1",
            "visit_id": "v1",
            "visit_date": "2026-08-02",
            "document_kind": "clinical_visit",
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапия",
            "filial": "1",
            "patient_id": "p1",
            "visit_time": "10:15",
            **_rich_clinical(),
            "mkb_code_main": "J03.9",
        },
        {
            "id": "2",
            "visit_id": "v2",
            "visit_date": "2026-08-02",
            "document_kind": "procedure_session",
            "doctor_fio": "Петров П.П.",
            "doctor_specialization": "УЗИ",
            "filial": "1",
            "patient_id": "p2",
        },
    ]
    cases = [
        {
            "mis_id": "1",
            "visit_id": "v1",
            "overall_pct": 80,
            "status": "good",
            "block_scores": {"exams": 70, "treatment": 65},
            "protocol_suggest": {
                "items": [
                    {
                        "title": "Острый тонзиллит",
                        "match_kind": "clinical",
                        "score": 80,
                        "protocol_id": "t1",
                    }
                ]
            },
            "evaluation_v4": {"scorer_version": "test", "schema_version": "1", "findings": []},
            **_rich_clinical(),
        },
        {"mis_id": "2", "visit_id": "v2", "overall_pct": None, "status": ""},
    ]
    report = {"date": "2026-08-02", "quality": {"passed": True}, "partial": False}
    upsert_warehouse(path, raw_rows, cases, report)
    import sqlite3

    with sqlite3.connect(path) as db:
        cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
        assert "zone1_pct" in cols
        assert "attention_primary" in cols
        clinical = db.execute(
            "SELECT zone1_band, zone2a_band, zone2b_kp_status, layer_engine "
            "FROM fact_mo_case WHERE mis_id='1'"
        ).fetchone()
        assert clinical[3] == "mo_zones_v1"
        assert clinical[0] in {"ok", "weak", "bad"}
        non = db.execute(
            "SELECT zone1_pct, attention_primary FROM fact_mo_case WHERE mis_id='2'"
        ).fetchone()
        assert non[0] is None
        assert non[1] in (None, "none")


def test_warehouse_zone_columns_helper() -> None:
    zones = compute_mo_zone_scores(
        {
            "clinical": _rich_clinical(),
            "meta": {"visit_date": "2026-08-02", "visit_time": "10:00", "diagnosis_code": "J03.9"},
            "document_kind": "clinical_visit",
        }
    )
    flat = warehouse_zone_columns(zones)
    assert "zone1_pct" in flat
    assert flat["layer_engine"] == "mo_zones_v1"
