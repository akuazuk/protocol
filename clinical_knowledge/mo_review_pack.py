"""Пакет разбора методиста: снимок МО + system/LLM + решение (Render SQLite).

См. docs/plans/2026-08-05-mo-methodist-review-pack-v1.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mo_backend import CRM_ROLES, _connect, _utc, build_case_detail
from .mo_case_document import (
    build_case_document_payload,
    load_case_source_row,
)
from .mo_llm_action_judge import load_llm_action_judge_for_case

VERDICT_TRIPLE = frozenset({"agree", "partial", "disagree", "unreviewed"})
PROTOCOL_RELEVANCE = frozenset({"relevant", "partial", "irrelevant", "unreviewed"})
TRAINING_ELIGIBILITY_POLICY = "mo_training_eligibility_v1"
SAVE_CONFLICT_ERRORS = frozenset(
    {
        "document_revision_conflict",
        "review_revision_conflict",
        "idempotency_payload_conflict",
        "evaluation_run_mismatch",
        "supersedes_pack_case_mismatch",
        "supersedes_pack_not_found",
    }
)

REVIEW_PACK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_review_pack (
  pack_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  visit_id TEXT NOT NULL,
  mis_id TEXT,
  patient_id TEXT,
  visit_date TEXT,
  doctor_fio TEXT,
  specialty TEXT,
  filial TEXT,
  clinical_json TEXT NOT NULL,
  system_json TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  training_use INTEGER NOT NULL DEFAULT 1,
  actor TEXT,
  created_at TEXT NOT NULL,
  supersedes_pack_id TEXT,
  document_revision TEXT,
  evaluation_run_id TEXT,
  idempotency_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_crm_review_pack_case
  ON crm_review_pack(case_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crm_review_pack_training
  ON crm_review_pack(training_use, created_at DESC);
"""


def ensure_review_pack_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    db = conn or _connect()
    try:
        db.executescript(REVIEW_PACK_SCHEMA_SQL)
        columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(crm_review_pack)").fetchall()
        }
        for name in (
            "document_revision",
            "evaluation_run_id",
            "idempotency_key",
            "request_hash",
        ):
            if name not in columns:
                db.execute(f"ALTER TABLE crm_review_pack ADD COLUMN {name} TEXT")
        if "review_revision" not in columns:
            db.execute("ALTER TABLE crm_review_pack ADD COLUMN review_revision INTEGER")
        db.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_crm_review_pack_idempotency
               ON crm_review_pack(case_id, actor, idempotency_key)
               WHERE idempotency_key IS NOT NULL AND idempotency_key != ''"""
        )
        if own:
            db.commit()
    finally:
        if own:
            db.close()


def _medical_exam_roots() -> list[Path]:
    from .mo_case_document import _medical_exam_roots as roots

    return list(roots())


def _secure_day_rows(day: str) -> list[dict[str, str]]:
    """Строки secure CSV за день (первый найденный файл)."""
    key = str(day or "").strip()[:10]
    if len(key) < 10:
        return []
    year, month = key[:4], key[5:7]
    for root in _medical_exam_roots():
        path = root / "secure_cases" / year / month / f"mo_{key}.csv"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except OSError:
            continue
    return []


def patient_id_map_for_day(day: str) -> dict[str, str]:
    """visit_id / mis_id → patient_id из secure CSV за день (для methodist+)."""
    out: dict[str, str] = {}
    for row in _secure_day_rows(day):
        patient = str(row.get("patient_id") or "").strip()
        if not patient:
            continue
        for field in ("visit_id", "id", "mis_id"):
            value = str(row.get(field) or "").strip()
            if value and value not in out:
                out[value] = patient
    return out


def visit_identity_map_for_day(day: str) -> dict[str, dict[str, str]]:
    """visit_id / mis_id → идентификаторы врача и пациента из secure CSV."""
    out: dict[str, dict[str, str]] = {}
    for row in _secure_day_rows(day):
        doctor_id = (
            str(row.get("doctor_id") or "").strip()
            or str(row.get("specialist_id_from_visit") or "").strip()
            or str(row.get("specialist_id") or "").strip()
        )
        payload = {
            "patient_id": str(row.get("patient_id") or "").strip(),
            "doctor_id": doctor_id,
            "doctor_fio": str(row.get("doctor_fio") or "").strip(),
            "specialty": str(
                row.get("doctor_specialization") or row.get("specialty") or ""
            ).strip(),
            "filial": str(row.get("filial") or "").strip(),
        }
        if not any(payload.values()):
            continue
        for field in ("visit_id", "id", "mis_id"):
            value = str(row.get(field) or "").strip()
            if value and value not in out:
                out[value] = payload
    return out


def _lookup_identity(
    case_id: str,
    *,
    visit_date: str | None = None,
    mis_id: str | None = None,
) -> dict[str, str]:
    day = str(visit_date or "").strip()[:10]
    if day:
        day_map = visit_identity_map_for_day(day)
        for key in (case_id, mis_id):
            needle = str(key or "").strip()
            if needle and needle in day_map:
                return dict(day_map[needle])
    try:
        row = load_case_source_row(case_id, visit_date=visit_date, mis_id=mis_id)
    except Exception:  # noqa: BLE001
        return {}
    if not row:
        return {}
    doctor_id = (
        str(row.get("doctor_id") or "").strip()
        or str(row.get("specialist_id_from_visit") or "").strip()
        or str(row.get("specialist_id") or "").strip()
    )
    return {
        "patient_id": str(row.get("patient_id") or "").strip(),
        "doctor_id": doctor_id,
        "doctor_fio": str(row.get("doctor_fio") or "").strip(),
        "specialty": str(row.get("doctor_specialization") or row.get("specialty") or "").strip(),
        "filial": str(row.get("filial") or "").strip(),
    }


def lookup_patient_id(
    case_id: str,
    *,
    visit_date: str | None = None,
    mis_id: str | None = None,
) -> str:
    return str(
        _lookup_identity(case_id, visit_date=visit_date, mis_id=mis_id).get("patient_id") or ""
    ).strip()


def lookup_doctor_id(
    case_id: str,
    *,
    visit_date: str | None = None,
    mis_id: str | None = None,
) -> str:
    return str(
        _lookup_identity(case_id, visit_date=visit_date, mis_id=mis_id).get("doctor_id") or ""
    ).strip()


def lookup_case_identity(
    case_id: str,
    *,
    visit_date: str | None = None,
    mis_id: str | None = None,
) -> dict[str, str]:
    """patient_id / doctor_id / doctor_fio / specialty / filial из secure CSV."""
    return _lookup_identity(case_id, visit_date=visit_date, mis_id=mis_id)


def enrich_rows_with_patient_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Добавить patient_id и doctor_id в список публичных строк (очередь / документы)."""
    by_day: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        day = str(row.get("date") or row.get("visit_date") or "")[:10]
        if len(day) >= 10 and day not in by_day:
            by_day[day] = visit_identity_map_for_day(day)
    for row in rows:
        day = str(row.get("date") or row.get("visit_date") or "")[:10]
        day_map = by_day.get(day) or {}
        identity: dict[str, str] = {}
        for key in (row.get("visit_id"), row.get("case_id"), row.get("mis_id"), row.get("id")):
            needle = str(key or "").strip()
            if needle and needle in day_map:
                identity = day_map[needle]
                break
        if not row.get("patient_id"):
            row["patient_id"] = identity.get("patient_id") or ""
        if not row.get("doctor_id"):
            row["doctor_id"] = identity.get("doctor_id") or ""
        # Подставить ФИО/филиал из CSV только если в витрине пусто / placeholder.
        doctor = str(row.get("doctor_fio") or row.get("doctor") or "").strip()
        if (not doctor or doctor == "Врач не указан") and identity.get("doctor_fio"):
            row["doctor_fio"] = identity["doctor_fio"]
            row["doctor"] = identity["doctor_fio"]
        branch = str(row.get("filial") or row.get("branch") or "").strip()
        if (not branch or branch == "Филиал не указан") and identity.get("filial"):
            row["filial"] = identity["filial"]
            row["branch"] = identity["filial"]
        specialty = str(row.get("specialty") or row.get("specialization") or "").strip()
        if (not specialty or specialty == "Специальность не указана") and identity.get("specialty"):
            row["specialty"] = identity["specialty"]
            row["specialization"] = identity["specialty"]
    return rows


def _normalize_decision(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {
        "status": str(data.get("status") or "in_review").strip()[:40],
        "assignee": str(data.get("assignee") or "").strip()[:120],
        "due_date": str(data.get("due_date") or "").strip()[:10],
        "tags": [str(t).strip()[:50] for t in (data.get("tags") or []) if str(t).strip()][:30],
        "finding_decisions": {},
        "verdict_completeness": "unreviewed",
        "verdict_diagnosis": "unreviewed",
        "verdict_recommendations": "unreviewed",
        "corrected_scores": {},
        "summary_ru": str(data.get("summary_ru") or data.get("comment") or "").strip()[:12000],
        "training_use": bool(data["training_use"]) if "training_use" in data else False,
        "protocol_ratings": [],
        "protocol_suggest": None,
    }
    for key in ("verdict_completeness", "verdict_diagnosis", "verdict_recommendations"):
        value = str(data.get(key) or "unreviewed").strip().lower()
        out[key] = value if value in VERDICT_TRIPLE else "unreviewed"
    # Поля % в UI сняты; старые клиенты всё ещё могут прислать scores.
    corrected = data.get("corrected_scores") if isinstance(data.get("corrected_scores"), dict) else {}
    for axis in ("completeness", "diagnosis", "recommendations"):
        try:
            n = int(round(float(corrected.get(axis))))
        except (TypeError, ValueError):
            continue
        if 0 <= n <= 100:
            out["corrected_scores"][axis] = n
    decisions = data.get("finding_decisions") if isinstance(data.get("finding_decisions"), dict) else {}
    for code, decision in decisions.items():
        d = str(decision).strip()
        if d in {"confirmed", "false_positive", "needs_more_data", "unreviewed"}:
            out["finding_decisions"][str(code)[:120]] = d
    ratings_raw = data.get("protocol_ratings")
    if isinstance(ratings_raw, list):
        for item in ratings_raw[:20]:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("protocol_id") or "").strip()[:160]
            if not pid:
                continue
            relevance = str(item.get("relevance") or "unreviewed").strip().lower()
            if relevance not in PROTOCOL_RELEVANCE:
                relevance = "unreviewed"
            out["protocol_ratings"].append(
                {
                    "protocol_id": pid,
                    "relevance": relevance,
                    "rank_ok": bool(item.get("rank_ok")) if item.get("rank_ok") is not None else None,
                    "note_ru": str(item.get("note_ru") or "").strip()[:500],
                    "title": str(item.get("title") or "").strip()[:240],
                }
            )
    suggest = data.get("protocol_suggest")
    if isinstance(suggest, dict) and suggest.get("items") is not None:
        out["protocol_suggest"] = {
            "engine": str(suggest.get("engine") or "case_protocol_suggest_v1")[:80],
            "generated_at": str(suggest.get("generated_at") or "")[:40],
            "items": list(suggest.get("items") or [])[:8],
        }
    return out


def _public_pack_row(row: sqlite3.Row | Mapping[str, Any], *, include_bodies: bool) -> dict[str, Any]:
    item = dict(row)
    base = {
        "pack_id": item.get("pack_id"),
        "case_id": item.get("case_id"),
        "visit_id": item.get("visit_id"),
        "mis_id": item.get("mis_id") or "",
        "patient_id": item.get("patient_id") or "",
        "visit_date": item.get("visit_date") or "",
        "doctor_fio": item.get("doctor_fio") or "",
        "specialty": item.get("specialty") or "",
        "filial": item.get("filial") or "",
        "training_use": bool(int(item.get("training_use") or 0)),
        "actor": item.get("actor") or "",
        "created_at": item.get("created_at") or "",
        "supersedes_pack_id": item.get("supersedes_pack_id") or None,
        "document_revision": item.get("document_revision"),
        "evaluation_run_id": item.get("evaluation_run_id") or None,
        "review_revision": item.get("review_revision"),
    }
    if not include_bodies:
        decision: dict[str, Any] = {}
        try:
            decision = json.loads(item.get("decision_json") or "{}")
        except json.JSONDecodeError:
            decision = {}
        base["decision_summary"] = {
            "status": decision.get("status"),
            "verdict_completeness": decision.get("verdict_completeness"),
            "verdict_diagnosis": decision.get("verdict_diagnosis"),
            "verdict_recommendations": decision.get("verdict_recommendations"),
            "summary_ru": (decision.get("summary_ru") or "")[:240],
            "training_use": bool(decision.get("training_use", False)),
            "review_revision": item.get("review_revision"),
        }
        return base
    for key, field in (
        ("clinical", "clinical_json"),
        ("system", "system_json"),
        ("decision", "decision_json"),
    ):
        try:
            base[key] = json.loads(item.get(field) or "{}")
        except json.JSONDecodeError:
            base[key] = {}
    return base


def canonical_review_request_hash(
    *,
    case_id: str,
    decision: Mapping[str, Any],
    expected_document_revision: str | int | None = None,
    expected_pack_id: str | None = None,
    expected_review_revision: str | int | None = None,
) -> str:
    payload = {
        "case_id": str(case_id or ""),
        "decision": dict(decision),
        "expected_document_revision": (
            str(expected_document_revision) if expected_document_revision is not None else ""
        ),
        "expected_pack_id": str(expected_pack_id or ""),
        "expected_review_revision": (
            str(expected_review_revision) if expected_review_revision is not None else ""
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_training_eligible(pack: Mapping[str, Any] | None) -> bool:
    item = dict(pack or {})
    decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
    if not item.get("training_use") and not decision.get("training_use"):
        return False
    eligibility = decision.get("training_eligibility")
    if isinstance(eligibility, dict) and eligibility.get("revoked_at"):
        return False
    return True


def _latest_review_row(conn: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT pack_id, review_revision, request_hash, evaluation_run_id, created_at
           FROM crm_review_pack
           WHERE case_id=?
           ORDER BY COALESCE(review_revision, 0) DESC, created_at DESC, pack_id DESC
           LIMIT 1""",
        (case_id,),
    ).fetchone()


def _replay_payload(existing: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    item = dict(existing)
    return {
        "ok": True,
        "pack_id": str(item["pack_id"]),
        "case_id": case_id,
        "created_at": str(item["created_at"]),
        "training_use": bool(item.get("training_use")),
        "supersedes_pack_id": item.get("supersedes_pack_id"),
        "review_revision": item.get("review_revision"),
        "idempotent_replay": True,
    }


def save_review_pack(
    *,
    case_id: str,
    actor: str,
    role: str,
    decision: dict[str, Any] | None,
    supersedes_pack_id: str | None = None,
    month: str | None = None,
    expected_document_revision: str | int | None = None,
    expected_pack_id: str | None = None,
    expected_review_revision: str | int | None = None,
    evaluation_run_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if role not in CRM_ROLES:
        raise PermissionError("mutation_requires_methodist_role")
    cid = str(case_id or "").strip()
    if not cid:
        raise ValueError("case_id_required")
    decision_norm = _normalize_decision(decision)
    if role == "expert":
        decision_norm["source"] = "expert"
        if not str(actor or "").startswith("expert:"):
            actor = f"expert:{actor}"
    else:
        decision_norm.setdefault("source", "methodist")
    idem = str(idempotency_key or "").strip()[:160]
    request_hash = canonical_review_request_hash(
        case_id=cid,
        decision=decision_norm,
        expected_document_revision=expected_document_revision,
        expected_pack_id=expected_pack_id,
        expected_review_revision=expected_review_revision,
    )
    detail = build_case_detail(cid, month=month)
    if not detail.get("ok"):
        raise ValueError("case_not_found")
    record = detail.get("record") if isinstance(detail.get("record"), dict) else {}
    assessment = (
        detail.get("assessment")
        if isinstance(detail.get("assessment"), dict)
        else (
            record.get("assessment")
            if isinstance(record.get("assessment"), dict)
            else {}
        )
    )
    current_revision = assessment.get("document_revision")
    if current_revision is None:
        current_revision = record.get("document_revision")
    if (
        expected_document_revision is not None
        and current_revision is not None
        and str(expected_document_revision) != str(current_revision)
    ):
        raise ValueError("document_revision_conflict")
    server_run_id = str(
        assessment.get("evaluation_run_id") or record.get("evaluation_run_id") or ""
    ).strip()
    client_run_id = str(evaluation_run_id or "").strip()
    if client_run_id and client_run_id != server_run_id:
        raise ValueError("evaluation_run_mismatch")
    current_run_id = server_run_id
    visit_date = str(record.get("date") or record.get("visit_date") or "")[:10]
    mis_id = str(record.get("mis_id") or "")
    patient_id = lookup_patient_id(cid, visit_date=visit_date or None, mis_id=mis_id or None)
    clinical: dict[str, Any] = {}
    try:
        document = build_case_document_payload(cid, month=month, detail=detail)
        if document.get("ok"):
            clinical = document.get("clinical") or {}
    except Exception:  # noqa: BLE001
        clinical = {}
    judge = load_llm_action_judge_for_case(cid, visit_date=visit_date)
    protocol_suggest = decision_norm.pop("protocol_suggest", None)
    if not isinstance(protocol_suggest, dict):
        try:
            from .case_protocol_suggest import suggest_protocols_for_mo_case

            if patient_id and not record.get("patient_id"):
                record = dict(record)
                record["patient_id"] = patient_id
            protocol_suggest = suggest_protocols_for_mo_case(
                clinical=clinical,
                record=record,
                findings=detail.get("findings") or [],
                llm_judge=judge if isinstance(judge, dict) else {},
                limit=3,
                attach_history=True,
            )
        except Exception:  # noqa: BLE001
            protocol_suggest = {"ok": False, "items": [], "available": False}
    system_snapshot = {
        "overall_pct": detail.get("deep_overall_pct")
        if detail.get("deep_overall_pct") is not None
        else record.get("overall_pct"),
        "status": detail.get("deep_status") or record.get("status"),
        "findings": detail.get("findings") or [],
        "axes": detail.get("axes") or {},
        "rubric_mz": detail.get("rubric_mz") or {},
        "llm_action_judge": judge,
        "protocol_suggest": protocol_suggest,
        "document_revision": current_revision,
        "evaluation_run_id": current_run_id or None,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    pack_id = str(uuid.uuid4())
    now = _utc()
    decision_norm["training_eligibility"] = {
        "eligible": bool(decision_norm.get("training_use")),
        "policy_version": TRAINING_ELIGIBILITY_POLICY,
        "decided_at": now,
        "decided_by": actor,
        "revoked_at": None,
    }
    with closing(_connect()) as conn:
        ensure_review_pack_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            if idem:
                existing = conn.execute(
                    """SELECT pack_id, created_at, training_use, supersedes_pack_id,
                              request_hash, review_revision
                       FROM crm_review_pack
                       WHERE case_id=? AND actor=? AND idempotency_key=?""",
                    (cid, actor, idem),
                ).fetchone()
                if existing:
                    stored_hash = str(existing["request_hash"] or "")
                    if stored_hash and stored_hash != request_hash:
                        raise ValueError("idempotency_payload_conflict")
                    replay = _replay_payload(existing, cid)
                    conn.commit()
                    return replay
            latest = _latest_review_row(conn, cid)
            latest_revision = int(latest["review_revision"] or 0) if latest else 0
            latest_pack_id = str(latest["pack_id"]) if latest else ""
            if expected_review_revision is not None and str(expected_review_revision) != str(
                latest_revision
            ):
                raise ValueError("review_revision_conflict")
            if expected_pack_id is not None and str(expected_pack_id or "") != latest_pack_id:
                raise ValueError("review_revision_conflict")
            if supersedes_pack_id:
                predecessor = conn.execute(
                    "SELECT pack_id, case_id FROM crm_review_pack WHERE pack_id=?",
                    (str(supersedes_pack_id),),
                ).fetchone()
                if not predecessor:
                    raise ValueError("supersedes_pack_not_found")
                if str(predecessor["case_id"]) != cid:
                    raise ValueError("supersedes_pack_case_mismatch")
            next_revision = latest_revision + 1
            inserted = conn.execute(
                """INSERT OR IGNORE INTO crm_review_pack(
                     pack_id, case_id, visit_id, mis_id, patient_id, visit_date,
                     doctor_fio, specialty, filial, clinical_json, system_json,
                     decision_json, training_use, actor, created_at, supersedes_pack_id,
                     document_revision, evaluation_run_id, idempotency_key,
                     request_hash, review_revision
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pack_id,
                    cid,
                    str(record.get("visit_id") or cid),
                    mis_id,
                    patient_id,
                    visit_date,
                    str(record.get("doctor_fio") or ""),
                    str(record.get("specialization") or record.get("specialty") or ""),
                    str(record.get("filial") or ""),
                    json.dumps(clinical, ensure_ascii=False),
                    json.dumps(system_snapshot, ensure_ascii=False),
                    json.dumps(decision_norm, ensure_ascii=False),
                    1 if decision_norm.get("training_use") else 0,
                    actor,
                    now,
                    str(supersedes_pack_id) if supersedes_pack_id else None,
                    str(current_revision) if current_revision is not None else None,
                    current_run_id or None,
                    idem or None,
                    request_hash,
                    next_revision,
                ),
            )
            if inserted.rowcount == 0:
                existing = conn.execute(
                    """SELECT pack_id, created_at, training_use, supersedes_pack_id,
                              request_hash, review_revision
                       FROM crm_review_pack
                       WHERE case_id=? AND actor=? AND idempotency_key=?""",
                    (cid, actor, idem),
                ).fetchone()
                if existing:
                    stored_hash = str(existing["request_hash"] or "")
                    if stored_hash and stored_hash != request_hash:
                        raise ValueError("idempotency_payload_conflict")
                    replay = _replay_payload(existing, cid)
                    conn.commit()
                    return replay
                raise sqlite3.IntegrityError("review_pack_insert_failed")
            tags_json = json.dumps(decision_norm.get("tags") or [], ensure_ascii=False)
            findings_json = json.dumps(decision_norm.get("finding_decisions") or {}, ensure_ascii=False)
            current_state = conn.execute(
                "SELECT case_id FROM crm_case_state WHERE case_id=?",
                (cid,),
            ).fetchone()
            if current_state:
                conn.execute(
                    """UPDATE crm_case_state
                       SET status=?, assignee=?, tags_json=?, due_date=?,
                           finding_decisions_json=?, updated_at=?, updated_by=?
                       WHERE case_id=?""",
                    (
                        decision_norm.get("status") or "in_review",
                        decision_norm.get("assignee") or None,
                        tags_json,
                        decision_norm.get("due_date") or None,
                        findings_json,
                        now,
                        actor,
                        cid,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO crm_case_state(
                         case_id, status, assignee, tags_json, due_date,
                         finding_decisions_json, updated_at, updated_by
                       ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        cid,
                        decision_norm.get("status") or "in_review",
                        decision_norm.get("assignee") or None,
                        tags_json,
                        decision_norm.get("due_date") or None,
                        findings_json,
                        now,
                        actor,
                    ),
                )
            conn.execute(
                """INSERT INTO crm_case_event(
                     event_id, case_id, event_type, actor, payload_json, created_at
                   ) VALUES (?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    cid,
                    "review_pack_saved",
                    actor,
                    json.dumps(
                        {
                            "pack_id": pack_id,
                            "review_revision": next_revision,
                            "request_hash": request_hash,
                            "training_use": decision_norm.get("training_use"),
                            "verdicts": {
                                "completeness": decision_norm.get("verdict_completeness"),
                                "diagnosis": decision_norm.get("verdict_diagnosis"),
                                "recommendations": decision_norm.get("verdict_recommendations"),
                            },
                            "summary_ru": (decision_norm.get("summary_ru") or "")[:500],
                        },
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {
        "ok": True,
        "pack_id": pack_id,
        "case_id": cid,
        "created_at": now,
        "patient_id": patient_id,
        "training_use": bool(decision_norm.get("training_use")),
        "supersedes_pack_id": str(supersedes_pack_id) if supersedes_pack_id else None,
        "document_revision": current_revision,
        "evaluation_run_id": current_run_id or None,
        "review_revision": next_revision,
        "request_hash": request_hash,
        "idempotent_replay": False,
    }


def list_review_packs(case_id: str, *, limit: int = 50) -> dict[str, Any]:
    cid = str(case_id or "").strip()
    if not cid:
        raise ValueError("case_id_required")
    with closing(_connect()) as conn:
        ensure_review_pack_schema(conn)
        rows = conn.execute(
            """SELECT pack_id, case_id, visit_id, mis_id, patient_id, visit_date,
                      doctor_fio, specialty, filial, decision_json, training_use,
                      actor, created_at, supersedes_pack_id,
                      document_revision, evaluation_run_id, review_revision
               FROM crm_review_pack
               WHERE case_id=?
               ORDER BY created_at DESC
               LIMIT ?""",
            (cid, max(1, min(int(limit), 200))),
        ).fetchall()
    return {
        "ok": True,
        "case_id": cid,
        "items": [_public_pack_row(row, include_bodies=False) for row in rows],
    }


def get_review_pack(pack_id: str) -> dict[str, Any]:
    pid = str(pack_id or "").strip()
    if not pid:
        raise ValueError("pack_id_required")
    with closing(_connect()) as conn:
        ensure_review_pack_schema(conn)
        row = conn.execute("SELECT * FROM crm_review_pack WHERE pack_id=?", (pid,)).fetchone()
    if not row:
        return {"ok": False, "error": "pack_not_found"}
    return {"ok": True, "pack": _public_pack_row(row, include_bodies=True)}


def revise_review_pack(
    *,
    pack_id: str,
    actor: str,
    role: str,
    decision: dict[str, Any] | None,
    month: str | None = None,
) -> dict[str, Any]:
    current = get_review_pack(pack_id)
    if not current.get("ok"):
        raise ValueError("pack_not_found")
    pack = current["pack"]
    base_decision = dict(pack.get("decision") or {})
    if isinstance(decision, dict):
        base_decision.update(decision)
    return save_review_pack(
        case_id=str(pack.get("case_id") or ""),
        actor=actor,
        role=role,
        decision=base_decision,
        supersedes_pack_id=str(pack.get("pack_id") or pack_id),
        expected_pack_id=str(pack.get("pack_id") or pack_id),
        expected_review_revision=pack.get("review_revision"),
        month=month,
    )


def revoke_training_eligibility(
    *,
    pack_id: str,
    actor: str,
    role: str,
    reason: str = "",
) -> dict[str, Any]:
    if role not in CRM_ROLES:
        raise PermissionError("mutation_requires_methodist_role")
    current = get_review_pack(pack_id)
    if not current.get("ok"):
        raise ValueError("pack_not_found")
    pack = current["pack"]
    decision = dict(pack.get("decision") or {})
    now = _utc()
    eligibility = dict(decision.get("training_eligibility") or {})
    eligibility.update(
        {
            "eligible": False,
            "policy_version": TRAINING_ELIGIBILITY_POLICY,
            "revoked_at": now,
            "revoked_by": actor,
            "revoke_reason": str(reason or "").strip()[:500],
        }
    )
    decision["training_use"] = False
    decision["training_eligibility"] = eligibility
    with closing(_connect()) as conn:
        ensure_review_pack_schema(conn)
        conn.execute(
            """UPDATE crm_review_pack
               SET training_use=0, decision_json=?
               WHERE pack_id=?""",
            (json.dumps(decision, ensure_ascii=False), str(pack_id)),
        )
        conn.commit()
    return {"ok": True, "pack_id": str(pack_id), "training_use": False, "revoked_at": now}
