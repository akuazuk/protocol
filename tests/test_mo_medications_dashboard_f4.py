"""F4: /medications-dashboard - типы, специальности, топ МНН, тренд, пары DDI."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from rag_server import api_methodist_mo_medications_dashboard
from test_mo_overview_dashboard_f1 import BASE, _seed


_PAIRS = [
    ("ибупрофен", "варфарин"),
    ("амиодарон", "дигоксин"),
    ("метотрексат", "ибупрофен"),
    ("симвастатин", "кларитромицин"),
    ("лизиноприл", "спиронолактон"),
    ("трамадол", "сертралин"),
    ("варфарин", "аспирин"),
    ("омепразол", "клопидогрел"),
    ("ципрофлоксацин", "тизанидин"),
    ("флуконазол", "аторвастатин"),
    ("метопролол", "верапамил"),
    ("калий", "спиронолактон"),
]


def _seed_meds(path: Path) -> None:
    _seed(path)
    extra: list[tuple[str, str, str, int, str, str, str]] = []
    for index, (left, right) in enumerate(_PAIRS):
        extra.append(
            (
                f"m08{index}",
                "C_ddi",
                "P1",
                0,
                f"Лекарственное взаимодействие (Major): {left} + {right}",
                f"{left} + {right}",
                f"{left} + {right}",
            )
        )
    for index in range(0, 40, 5):
        extra.append((f"m08{index}", "C_nsaid_dup", "P1", 0, "Одновременно ≥2 НПВП", "ибупрофен, диклофенак", "ибупрофен, диклофенак"))
    for index in range(0, 40, 6):
        extra.append((f"m08{index}", "C_rceth_off_label", "P2", 0, "Вне инструкции: омепразол", "омепразол", "омепразол"))
    for index in range(0, 40, 8):
        extra.append((f"m08{index}", "B_tx_offprotocol", "P2", 0, "Назначение не по протоколу", "амоксициллин", "амоксициллин"))
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO fact_mo_finding
               (mis_id,finding_code,severity,passed,title_ru,detail_ru,evidence)
               VALUES(?,?,?,?,?,?,?)""",
            extra,
        )
        conn.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "meds_dash.sqlite"
    _seed_meds(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


def test_endpoint_declares_period_and_facets() -> None:
    names = set(inspect.signature(api_methodist_mo_medications_dashboard).parameters)
    for key in ("period", "date_from", "date_to", "specializations", "filials", "doctors", "finding_codes"):
        assert key in names


def test_blocks_present_and_types_cover_tiles(warehouse: Path) -> None:
    out = mo_backend.build_medications_dashboard({**BASE, "document_kinds": "clinical_visit"})
    assert out["ok"] and out["available"]
    assert out["total_cases"] == 40
    assert [t["id"] for t in out["types"]] == ["interactions", "duplicates", "dose_label", "offprotocol"]
    by_id = {t["id"]: t for t in out["types"]}
    assert by_id["interactions"]["n"] >= 12
    assert by_id["duplicates"]["n"] >= 1
    assert by_id["dose_label"]["n"] >= 1
    assert by_id["offprotocol"]["n"] >= 1
    assert all(t.get("precision_note") for t in out["types"])
    assert {t["id"] for t in out["tiles"]} >= {"any", "interactions"}
    inns = [d["inn"] for d in out["drugs"]]
    assert "ибупрофен" in inns
    assert out["trend"]
    assert {r["specialty"] for r in out["specialty"]["rows"]} >= {"Терапия", "Кардиология"}
    assert any((r["per_100"]["interactions"] or 0) > 0 for r in out["specialty"]["rows"])
    assert out["pairs"]["available"] is True
    assert len(out["pairs"]["items"]) >= 10
    assert out["pairs"]["nodes"] and out["pairs"]["links"]


def test_facets_apply_to_every_block(warehouse: Path) -> None:
    full = mo_backend.build_medications_dashboard({**BASE, "document_kinds": "clinical_visit"})
    part = mo_backend.build_medications_dashboard(
        {**BASE, "document_kinds": "clinical_visit", "specializations": "Терапия"}
    )
    assert part["total_cases"] == 20 < full["total_cases"]
    assert [r["specialty"] for r in part["specialty"]["rows"]] == ["Терапия"]
    assert sum(t["n"] for t in part["types"]) < sum(t["n"] for t in full["types"])
    by_doctor = mo_backend.build_medications_dashboard(
        {**BASE, "document_kinds": "clinical_visit", "doctors": "Врач Б"}
    )
    assert by_doctor["total_cases"] == 20


def test_pairs_hidden_below_threshold(warehouse: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mo_backend, "_MED_PAIR_MIN", 100)
    out = mo_backend.build_medications_dashboard({**BASE, "document_kinds": "clinical_visit"})
    assert out["pairs"]["available"] is False
    assert out["pairs"]["items"]
    assert out["pairs"]["nodes"] == []
    assert out["pairs"]["reason"]


def test_empty_window_has_reason(warehouse: Path) -> None:
    out = mo_backend.build_medications_dashboard(
        {"period": "custom", "date_from": "2025-01-01", "date_to": "2025-01-31"}
    )
    assert out["ok"]
    assert out["available"] is False
    assert out["reason"]
    assert out["types"] == []
