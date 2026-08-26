"""Shadow lab reconcile: exam_recommendations vs type_name, no primary score."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import patient_key_for
from clinical_knowledge.mo_lab_bundle import lab_payload_for_case
from clinical_knowledge.mo_lab_shadow import (
    CODE_ORDERED_DONE,
    CODE_PRESENT_GAP,
    apply_lab_to_result,
    build_lab_reconcile,
    evaluate_lab_for_case,
    lab_shadow_findings,
    ordered_panels,
)

DDL = """
CREATE TABLE fact_mo_lab (
  patient_key TEXT NOT NULL,
  test_date TEXT NOT NULL,
  test_id INTEGER NOT NULL,
  type_id INTEGER,
  type_name TEXT,
  indicator_id INTEGER,
  indicator_name TEXT,
  value TEXT,
  unit TEXT
);
"""


def _seed(path: Path) -> None:
    pk = patient_key_for("1001")
    db = sqlite3.connect(path)
    db.executescript(DDL)
    rows = [
        (pk, "2026-08-10", 1, 10, "Общий анализ крови", 101, "Гемоглобин", "132", "г/л"),
        (pk, "2026-08-20", 2, 20, "Биохимический анализ крови", 201, "Глюкоза", "5.4", "ммоль/л"),
        (patient_key_for("9999"), "2026-08-20", 4, 10, "Общий анализ крови", 101, "Чужой", "1", ""),
    ]
    db.executemany("INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()


def test_ordered_oak_already_on_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    payload = lab_payload_for_case(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    recon = build_lab_reconcile(
        payload,
        {"exam_recommendations": "Контроль ОАК, глюкоза крови", "exam_data": ""},
    )
    labels_present = {row["label"] for row in recon["present"]}
    assert "ОАК" in labels_present
    assert "БАК" in labels_present
    done = {row["label"] for row in recon["ordered_and_present"]}
    assert "ОАК" in done
    assert "глюкоза" in done
    assert "глюкоза" not in {
        row["label"] for row in recon["ordered_not_in_warehouse"]
    }
    findings = lab_shadow_findings(recon)
    codes = {f["code"] for f in findings}
    assert CODE_ORDERED_DONE in codes
    assert all(f.get("shadow") and f.get("is_shadow") for f in findings)
    assert "132" not in str(findings)


def test_present_not_written_in_mo(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    payload = lab_payload_for_case(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    recon = build_lab_reconcile(
        payload,
        {"exam_recommendations": "УЗИ ОБП", "exam_data": "без патологии"},
    )
    gap = {row["label"] for row in recon["present_not_in_mo"]}
    assert "ОАК" in gap
    assert "БАК" in gap
    codes = {f["code"] for f in lab_shadow_findings(recon)}
    assert CODE_PRESENT_GAP in codes
    assert CODE_ORDERED_DONE not in codes


def test_no_finding_when_ordered_missing_from_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    payload = lab_payload_for_case(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    recon = build_lab_reconcile(
        payload,
        {"exam_recommendations": "ПСА, ТТГ", "exam_data": "ОАК от 10.08, биохимия крови"},
    )
    missing = {row["label"] for row in recon["ordered_not_in_warehouse"]}
    assert "ПСА" in missing
    assert "ТТГ" in missing
    codes = {f["code"] for f in lab_shadow_findings(recon)}
    assert CODE_ORDERED_DONE not in codes
    assert CODE_PRESENT_GAP not in codes


def test_skip_gap_if_exam_slots_not_loaded(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    payload = lab_payload_for_case(
        {"patient_id": "1001", "visit_date": "2026-08-20"},
        lab_db=db,
    )
    recon = build_lab_reconcile(payload, {"patient_id": "1001"})
    assert recon["present_not_in_mo"] == []
    assert lab_shadow_findings(recon) == []


def test_bak_token_does_not_hit_bacteriology() -> None:
    ordered = ordered_panels("бак посев мочи, бактериологическое исследование")
    assert "bak" not in ordered


def test_apply_does_not_touch_primary_by_default(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    monkeypatch.delenv("MO_LAB_IN_PRIMARY", raising=False)
    result = {
        "findings": [{"code": "B_dx_absent", "severity": "P1", "shadow": False}],
    }
    apply_lab_to_result(
        result,
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "exam_recommendations": "ОАК",
            "exam_data": "",
        },
        lab_db=db,
    )
    shadows = [f for f in result["findings"] if f.get("code", "").startswith("B_lab_")]
    assert shadows and all(f.get("is_shadow") for f in shadows)
    assert any(f.get("code") == "B_dx_absent" and not f.get("is_shadow") for f in result["findings"])


def test_primary_flag_promotes_gap_only(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_LAB_IN_PRIMARY", "1")
    monkeypatch.setattr(
        "clinical_knowledge.mo_lab_rollout.lab_primary_guard",
        lambda: {"effective": True},
    )
    result = {"findings": []}
    apply_lab_to_result(
        result,
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "exam_recommendations": "ОАК",
            "exam_data": "",
        },
        lab_db=db,
    )
    by_code = {f["code"]: f for f in result["findings"] if str(f.get("code") or "").startswith("B_lab_")}
    assert by_code[CODE_ORDERED_DONE].get("is_shadow") is True
    assert by_code[CODE_PRESENT_GAP].get("is_shadow") is False
    assert "132" not in str(result["findings"])


def test_primary_flag_keeps_prior_only_gap_in_shadow(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_LAB_IN_PRIMARY", "1")
    monkeypatch.setattr(
        "clinical_knowledge.mo_lab_rollout.lab_primary_guard",
        lambda: {"effective": True},
    )
    result = {"findings": []}
    apply_lab_to_result(
        result,
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "exam_recommendations": "",
            "exam_data": "Биохимический анализ крови",
        },
        lab_db=db,
    )
    gap = [
        item for item in result["findings"]
        if item.get("code") == CODE_PRESENT_GAP
    ]
    assert gap
    assert all(item.get("is_shadow") is True for item in gap)
    assert "ОАК" in str(gap)


def test_post_visit_result_never_creates_gap(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            (
                patient_key_for("1001"),
                "2026-08-21",
                5,
                30,
                "Общий анализ мочи",
                301,
                "Белок",
                "нет",
                "",
            ),
        )
    monkeypatch.setenv("MO_LAB_IN_PRIMARY", "1")
    payload, findings = evaluate_lab_for_case(
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "exam_recommendations": "",
            "exam_data": "",
        },
        lab_db=db,
    )
    post_labels = {
        row["label"] for row in payload["reconcile"]["post_visit_present"]
    }
    assert "ОАМ" in post_labels
    gap_details = " ".join(
        str(item.get("detail_ru") or "")
        for item in findings
        if item.get("code") == CODE_PRESENT_GAP
    )
    assert "ОАМ" not in gap_details


def test_reconcile_uses_full_index_beyond_display_cap(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    pk = patient_key_for("1001")
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    pk,
                    "2026-08-20",
                    1000 + idx,
                    900,
                    "Анализ без панели",
                    1000 + idx,
                    f"Показатель {idx:03d}",
                    str(idx),
                    "",
                )
                for idx in range(401)
            ],
        )
        conn.execute(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            (
                pk,
                "2026-08-10",
                2000,
                31,
                "Общий анализ мочи",
                2001,
                "Белок",
                "нет",
                "",
            ),
        )
    payload, _ = evaluate_lab_for_case(
        {
            "patient_id": "1001",
            "visit_date": "2026-08-20",
            "exam_recommendations": "ОАМ",
            "exam_data": "",
        },
        lab_db=db,
    )
    assert payload["summary"]["truncated"] is True
    assert "ОАМ" in {
        row["label"] for row in payload["reconcile"]["ordered_and_present"]
    }
