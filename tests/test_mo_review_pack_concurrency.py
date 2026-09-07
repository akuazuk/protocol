"""P0 save lineage: concurrent review, idempotency hash, training eligibility."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinical_knowledge import mo_review_pack
from clinical_knowledge.mo_daily import initialize_warehouse


def _seed_case(
    db: Path,
    *,
    visit_id: str = "3646270",
    mis_id: str = "898517",
    day: str = "2026-08-04",
    run_id: str = "run-case-a",
    revision: int = 3,
) -> None:
    initialize_warehouse(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fact_mo_case(
                 mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                 scorer_version, score_schema_version, doctor_key, specialty, filial,
                 diagnosis_code, icd_chapter, content_hash, updated_at,
                 evaluation_run_id, document_revision
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mis_id,
                visit_id,
                day,
                "clinical_visit",
                62.0,
                "review",
                "v3",
                "1",
                "doc-1",
                "Терапия",
                "Филиал Центр",
                "J06.9",
                "X",
                "hash",
                "2026-08-04T12:00:00Z",
                run_id,
                revision,
            ),
        )
        conn.execute(
            """INSERT OR REPLACE INTO dim_doctor(doctor_key, doctor_fio, specialty, filial)
               VALUES (?,?,?,?)""",
            ("doc-1", "Иванов И.И.", "Терапия", "Филиал Центр"),
        )
        conn.commit()


def _save(case_id: str = "3646270", **kwargs):
    decision = {
        "status": "confirmed_issue",
        "summary_ru": kwargs.pop("summary_ru", "Первое решение методиста"),
        "training_use": kwargs.pop("training_use", False),
    }
    decision.update(kwargs.pop("decision", {}))
    return mo_review_pack.save_review_pack(
        case_id=case_id,
        actor=kwargs.pop("actor", "Методист А"),
        role=kwargs.pop("role", "methodist"),
        decision=decision,
        **kwargs,
    )


def test_second_client_gets_conflict_and_first_pack_stays(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db)

    first = _save(expected_review_revision=0, summary_ru="Решение первого клиента")
    assert first["ok"] is True
    assert first["review_revision"] == 1

    with pytest.raises(ValueError, match="review_revision_conflict"):
        _save(
            actor="Методист Б",
            expected_review_revision=0,
            summary_ru="Поздняя попытка второго клиента",
        )

    listed = mo_review_pack.list_review_packs("3646270")
    assert len(listed["items"]) == 1
    full = mo_review_pack.get_review_pack(first["pack_id"])["pack"]
    assert full["decision"]["summary_ru"] == "Решение первого клиента"


def test_same_idempotency_key_replays_only_matching_hash(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db)

    first = _save(
        idempotency_key="save-1",
        expected_review_revision=0,
        summary_ru="Одинаковый запрос",
    )
    replay = _save(
        idempotency_key="save-1",
        expected_review_revision=0,
        summary_ru="Одинаковый запрос",
    )
    assert replay["idempotent_replay"] is True
    assert replay["pack_id"] == first["pack_id"]

    with pytest.raises(ValueError, match="idempotency_payload_conflict"):
        _save(
            idempotency_key="save-1",
            expected_review_revision=0,
            summary_ru="Другой payload с тем же ключом",
        )
    listed = mo_review_pack.list_review_packs("3646270")
    assert len(listed["items"]) == 1


def test_foreign_run_and_other_case_supersede_are_rejected(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db, visit_id="3646270", mis_id="898517", run_id="run-case-a")
    _seed_case(db, visit_id="3646271", mis_id="898518", run_id="run-case-b")

    other = _save(case_id="3646271", expected_review_revision=0, summary_ru="Чужой случай")
    with pytest.raises(ValueError, match="evaluation_run_mismatch"):
        _save(case_id="3646270", evaluation_run_id="run-case-b", expected_review_revision=0)
    with pytest.raises(ValueError, match="supersedes_pack_case_mismatch"):
        _save(
            case_id="3646270",
            expected_review_revision=0,
            supersedes_pack_id=other["pack_id"],
        )


def test_expert_false_training_use_is_kept_and_revoke_excludes_export(
    monkeypatch, tmp_path: Path
) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db)

    saved = _save(role="expert", actor="expert:anna", training_use=False, expected_review_revision=0)
    pack = mo_review_pack.get_review_pack(saved["pack_id"])["pack"]
    assert pack["decision"]["training_use"] is False
    assert pack["decision"]["training_eligibility"]["eligible"] is False
    assert mo_review_pack.is_training_eligible(pack) is False

    allowed = _save(
        actor="Методист",
        training_use=True,
        expected_review_revision=1,
        expected_pack_id=saved["pack_id"],
        summary_ru="Допуск на обучение",
    )
    allowed_pack = mo_review_pack.get_review_pack(allowed["pack_id"])["pack"]
    assert mo_review_pack.is_training_eligible(allowed_pack) is True
    revoked = mo_review_pack.revoke_training_eligibility(
        pack_id=allowed["pack_id"],
        actor="Методист",
        role="methodist",
        reason="отзыв согласия",
    )
    assert revoked["training_use"] is False
    after = mo_review_pack.get_review_pack(allowed["pack_id"])["pack"]
    assert mo_review_pack.is_training_eligible(after) is False


def test_http_rbac_and_conflict_status(monkeypatch, tmp_path: Path) -> None:
    import rag_server

    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("METHODIST_TOKEN", "synthetic-test-token")
    _seed_case(db)

    methodist = TestClient(
        rag_server.app,
        headers={"X-Methodist-Token": "synthetic-test-token", "X-Methodist-Role": "methodist"},
    )
    viewer = TestClient(
        rag_server.app,
        headers={"X-Methodist-Token": "synthetic-test-token", "X-Methodist-Role": "viewer"},
    )
    denied = viewer.post(
        "/api/methodist/mo/cases/3646270/review-pack",
        json={"decision": {"status": "in_review", "summary_ru": "viewer"}},
    )
    assert denied.status_code == 403

    first = methodist.post(
        "/api/methodist/mo/cases/3646270/review-pack",
        headers={"Idempotency-Key": "http-1"},
        json={
            "decision": {"status": "confirmed_issue", "summary_ru": "HTTP первое", "training_use": False},
            "expected_review_revision": 0,
        },
    )
    assert first.status_code == 200
    replay = methodist.post(
        "/api/methodist/mo/cases/3646270/review-pack",
        headers={"Idempotency-Key": "http-1"},
        json={
            "decision": {"status": "confirmed_issue", "summary_ru": "HTTP первое", "training_use": False},
            "expected_review_revision": 0,
        },
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["pack_id"] == first.json()["pack_id"]

    conflict = methodist.post(
        "/api/methodist/mo/cases/3646270/review-pack",
        headers={"Idempotency-Key": "http-1"},
        json={
            "decision": {"status": "confirmed_issue", "summary_ru": "HTTP другой", "training_use": False},
            "expected_review_revision": 0,
        },
    )
    assert conflict.status_code == 409
    stale = methodist.post(
        "/api/methodist/mo/cases/3646270/review-pack",
        headers={"Idempotency-Key": "http-2"},
        json={
            "decision": {"status": "in_review", "summary_ru": "устаревшая ревизия", "training_use": False},
            "expected_review_revision": 0,
        },
    )
    assert stale.status_code == 409
    listed = methodist.get("/api/methodist/mo/cases/3646270/review-packs")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1
