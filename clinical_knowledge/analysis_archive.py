"""Опциональный архив обезличенных снимков анализа КЗ (только manifest.jsonl).

По умолчанию выключен (CONSULT_ARCHIVE_ANALYSES=0). На Render можно включить - файл копится на диске сервиса. Для улучшения кода эталоны хранятся в git:
tests/fixtures/consult_replay.jsonl + clients_consult/*.pdf → git pull в Cursor → replay.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .privacy import name_to_initials

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "output" / "consult_archive"
_MANIFEST = "manifest.jsonl"


def archive_dir() -> Path:
    raw = os.environ.get("CONSULT_ARCHIVE_DIR", "").strip()
    return Path(raw) if raw else _DEFAULT_DIR


def archive_enabled() -> bool:
    return os.environ.get("CONSULT_ARCHIVE_ANALYSES", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_snapshot(
    *,
    full_text: str,
    source_file: str = "",
    build_version: str = "",
    structured_analysis: dict[str, Any] | None,
    review: dict[str, Any] | None = None,
    retrieval_paths: list[str] | None = None,
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Собирает обезличенный снимок (без полного текста КЗ)."""
    doc = (structured_analysis or {}).get("document") or {}
    comp = (structured_analysis or {}).get("compliance") or {}
    patient = doc.get("patient") or {}
    bd = comp.get("score_breakdown") or {}
    rubric = (structured_analysis or {}).get("rubric_specifics") or {}

    diags = []
    for d in comp.get("diagnosis_assessments") or []:
        diags.append({
            "icd10_code": d.get("icd10_code"),
            "status": d.get("status"),
            "text_len": len(d.get("diagnosis_text") or ""),
        })

    rag_score = None
    if isinstance(review, dict):
        rag_score = review.get("overall_compliance_pct")
        if rag_score is None and isinstance(review.get("summary"), dict):
            rag_score = review["summary"].get("overall_compliance_pct")

    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "build_version": build_version,
        "source_basename": Path(source_file).name if source_file else "",
        "text_hash": _text_hash(full_text),
        "text_length": len(full_text or ""),
        "patient_initials": name_to_initials(patient.get("full_name")),
        "age_years": patient.get("age_years"),
        "sex": patient.get("sex"),
        "doctor_specialty": doc.get("doctor_specialty"),
        "icd_codes": icd_codes or [],
        "rubric_slugs": rubric.get("rubrics") or [],
        "structured_overall_status": comp.get("overall_status"),
        "structured_overall_score": comp.get("overall_score"),
        "rag_overall_pct": rag_score,
        "score_breakdown": bd,
        "matched_protocol_paths": list(retrieval_paths or [])[:12],
        "diagnosis_assessments": diags,
        "safety_count": len(comp.get("safety_assessments") or []),
    }


def save_snapshot(snapshot: dict[str, Any]) -> Path | None:
    """Добавляет строку в manifest.jsonl (если архив включён)."""
    if not archive_enabled():
        return None
    try:
        d = archive_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / _MANIFEST
        line = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path
    except OSError:
        return None


def load_snapshots(limit: int | None = None) -> list[dict[str, Any]]:
    path = archive_dir() / _MANIFEST
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        return out[-limit:]
    return out
