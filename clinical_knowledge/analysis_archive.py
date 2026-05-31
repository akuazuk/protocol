"""Архив анонимизированных снимков анализа КЗ для регрессий и улучшения системы.

Каждый consult-review (при CONSULT_ARCHIVE_ANALYSES=1) сохраняет JSONL-запись без
полного текста КЗ — только хеш, инициалы, МКБ, баллы и подобранные протоколы.

Периодически (CONSULT_ARCHIVE_EXPORT_EVERY) выполняется автоэкспорт в
output/consult_archive/exports/latest.jsonl для скачивания и передачи в Cursor.
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
_EXPORTS = "exports"
_EXPORT_LATEST = "latest.jsonl"
_EXPORT_META = "export_meta.json"
_CURSOR_README = "CURSOR_README.txt"

_CURSOR_README_TEXT = """\
# Как использовать экспорт для улучшения системы (Cursor)

1. Скачайте latest.jsonl:
   GET /api/consult-archive/export/latest

2. Положите в репозиторий (опционально):
   cp latest.jsonl tests/fixtures/consult_replay.jsonl

3. Добавьте PDF-кейсы в clients_consult/ (если есть локально).

4. Запустите регрессию:
   python scripts/replay_consult_archive.py --fixtures tests/fixtures/consult_replay.jsonl

5. В Cursor откройте чат и опишите расхождения:
   «По replay_consult_archive DIFF на pl_1_f.pdf — исправь парсер/матчер,
    добавь регресс-тест».

Это не дообучение LLM, а накопление эталонных метрик + правка детерминированного кода.
Чем больше снимков — тем стабильнее подбор протоколов и разбор КЗ.
"""


def archive_dir() -> Path:
    raw = os.environ.get("CONSULT_ARCHIVE_DIR", "").strip()
    return Path(raw) if raw else _DEFAULT_DIR


def archive_enabled() -> bool:
    return os.environ.get("CONSULT_ARCHIVE_ANALYSES", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def export_every_n() -> int:
    try:
        return max(1, int(os.environ.get("CONSULT_ARCHIVE_EXPORT_EVERY", "5")))
    except ValueError:
        return 5


def export_batch_size() -> int:
    try:
        return max(5, int(os.environ.get("CONSULT_ARCHIVE_EXPORT_SIZE", "50")))
    except ValueError:
        return 50


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
    """Собирает обезличенный снимок результата анализа."""
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


def _manifest_path() -> Path:
    return archive_dir() / _MANIFEST


def _export_dir() -> Path:
    return archive_dir() / _EXPORTS


def manifest_count() -> int:
    path = _manifest_path()
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def save_snapshot(snapshot: dict[str, Any]) -> Path | None:
    """Добавляет снимок в manifest.jsonl. Возвращает путь или None при ошибке."""
    if not archive_enabled():
        return None
    try:
        d = archive_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = _manifest_path()
        line = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path
    except OSError:
        return None


def get_last_export_meta() -> dict[str, Any]:
    """Последний meta автоэкспорта (если был)."""
    meta_path = _export_dir() / _EXPORT_META
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def maybe_auto_export(*, build_version: str = "") -> dict[str, Any]:
    """Экспортирует последние N снимков каждые M записей в manifest."""
    if not archive_enabled():
        return {"enabled": False}

    total = manifest_count()
    last = get_last_export_meta()
    every = export_every_n()

    if total == 0 or total % every != 0:
        out = dict(last)
        out.setdefault("enabled", True)
        out["total_snapshots"] = total
        out["just_exported"] = False
        out["next_export_at"] = ((total // every) + 1) * every if total else every
        out["export_every"] = every
        if out.get("message_ru"):
            out["status_ru"] = out["message_ru"]
        elif total:
            left = out["next_export_at"] - total
            out["status_ru"] = (
                f"Снимок #{total} сохранён. Автоэкспорт через {left} "
                f"{'анализ' if left == 1 else 'анализа'}."
            )
        else:
            out["status_ru"] = "Архив анализов включён."
        return out

    batch_size = export_batch_size()
    snaps = load_snapshots(limit=batch_size)
    export_dir = _export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    latest = export_dir / _EXPORT_LATEST
    meta_path = export_dir / _EXPORT_META
    readme = export_dir / _CURSOR_README

    try:
        with latest.open("w", encoding="utf-8") as f:
            for s in snaps:
                f.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
        readme.write_text(_CURSOR_README_TEXT, encoding="utf-8")
    except OSError as exc:
        return {
            "enabled": True,
            "just_exported": False,
            "error": str(exc)[:200],
            "status_ru": "Ошибка автоэкспорта архива.",
        }

    meta = {
        "enabled": True,
        "just_exported": True,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "auto",
        "total_snapshots": total,
        "exported_count": len(snaps),
        "export_every": every,
        "export_path": str(latest),
        "download_api": "/api/consult-archive/export/latest",
        "readme_api": "/api/consult-archive/export/readme",
        "build_version": build_version or (snaps[-1].get("build_version") if snaps else ""),
        "message_ru": (
            f"Автоэкспорт выполнен: {len(snaps)} обезличенных снимков "
            f"(всего в архиве: {total})."
        ),
        "status_ru": (
            f"Автоэкспорт выполнен: {len(snaps)} снимков готовы для Cursor "
            f"(скачать → latest.jsonl)."
        ),
        "cursor_steps_ru": [
            "Скачайте latest.jsonl по ссылке ниже.",
            "Положите в tests/fixtures/consult_replay.jsonl (git).",
            "Запустите: python scripts/replay_consult_archive.py --fixtures …",
            "В Cursor опишите расхождения — правка парсера/матчера, не LLM.",
        ],
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return meta


def save_snapshot_with_export(
    snapshot: dict[str, Any],
    *,
    build_version: str = "",
) -> dict[str, Any]:
    """Сохраняет снимок и возвращает статус архива + автоэкспорта для UI."""
    path = save_snapshot(snapshot)
    total = manifest_count()
    export_meta = maybe_auto_export(build_version=build_version)
    return {
        "enabled": archive_enabled(),
        "saved": path is not None,
        "manifest_path": str(path) if path else None,
        "total_snapshots": total,
        "snapshot_hash": snapshot.get("text_hash"),
        **export_meta,
    }


def load_snapshots(limit: int | None = None) -> list[dict[str, Any]]:
    """Читает все снимки из manifest.jsonl."""
    path = _manifest_path()
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


def export_latest_path() -> Path:
    return _export_dir() / _EXPORT_LATEST


def export_readme_path() -> Path:
    return _export_dir() / _CURSOR_README
