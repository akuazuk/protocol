"""Prepare sectioned source text per PDF for LLM / structured extraction."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"
RICH = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
CATALOG = ROOT / "data" / "protocol_catalog.jsonl"
SOURCE_DIR = ROOT / "data" / "protocol_summaries" / "source_text"

_SECTION_HINTS = (
    ("classification", re.compile(r"классификац|диагноз|нозолог|мкб|icd", re.I)),
    ("diagnostics", re.compile(r"диагностик|обследован|лаборатор|инструментал", re.I)),
    ("treatment", re.compile(r"лечени|терапи|медикамент|фармак|хирург", re.I)),
    ("prevention", re.compile(r"профилактик|наблюден|диспансер", re.I)),
    ("routing", re.compile(r"госпитализац|маршрут|показан|направлен", re.I)),
    ("criteria", re.compile(r"критери|определени", re.I)),
    ("other", re.compile(r".*", re.I)),
)


def _protocol_id_from_path(path: str) -> str:
    from .builder import _protocol_id_from_path as _pid

    return _pid(path)


def _guess_section(title: str, chunk_type: str = "") -> str:
    ct = (chunk_type or "").strip().lower()
    if ct in ("classification", "diagnostics", "treatment", "drug_list", "criteria"):
        return ct if ct != "drug_list" else "treatment"
    t = (title or "").strip()
    for name, rx in _SECTION_HINTS:
        if rx.search(t):
            return name
    return "other"


def _load_chunks_for_path(path: str) -> list[dict[str, Any]]:
    norm = path.replace("\\", "/").strip()
    rich_path = RICH if RICH.is_file() else None
    plain_path = CHUNKS if CHUNKS.is_file() else None
    rows: list[dict[str, Any]] = []
    for fp in (rich_path, plain_path):
        if not fp:
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ch = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp = str(ch.get("source_path") or ch.get("path") or "").replace("\\", "/")
                if sp == norm or sp.endswith(norm.split("/")[-1]):
                    rows.append(ch)
        if rows:
            break
    return rows


def _catalog_entry(path: str) -> dict[str, Any]:
    if not CATALOG.is_file():
        return {}
    norm = path.replace("\\", "/")
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("path") or "").replace("\\", "/") == norm:
                return row
    return {}


def build_source_text_document(path: str) -> dict[str, Any]:
    """Sectioned document for one PDF."""
    chunks = _load_chunks_for_path(path)
    catalog = _catalog_entry(path)
    sections: dict[str, list[dict[str, Any]]] = {
        "classification": [],
        "diagnostics": [],
        "treatment": [],
        "prevention": [],
        "routing": [],
        "criteria": [],
        "other": [],
    }
    for ch in chunks:
        title = str(ch.get("section_title") or ch.get("title") or "")
        ctype = str(ch.get("chunk_type") or ch.get("kind") or "")
        sec = _guess_section(title, ctype)
        text = (ch.get("text") or "").strip()
        if not text or len(text) < 20:
            continue
        sections[sec].append(
            {
                "section_title": title[:200] or None,
                "chunk_type": ctype or None,
                "page_from": ch.get("page_from") or ch.get("page"),
                "page_to": ch.get("page_to"),
                "text": text[:4000],
            }
        )
    for k in sections:
        sections[k] = sections[k][:12]
    title = catalog.get("display_title_normalized") or catalog.get("display_title") or Path(path).stem
    return {
        "path": path.replace("\\", "/"),
        "protocol_id": _protocol_id_from_path(path),
        "title": str(title)[:300],
        "specialty_slug": catalog.get("specialty_slug"),
        "audience": catalog.get("audience"),
        "icd10_primary": list(catalog.get("icd10_primary") or [])[:20],
        "icd10_all": list(catalog.get("icd10_all") or [])[:40],
        "sections": sections,
        "chunk_count": len(chunks),
    }


def save_source_text(doc: dict[str, Any]) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pid = doc.get("protocol_id") or "unknown"
    out = SOURCE_DIR / f"{pid}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_source_text(protocol_id: str) -> dict[str, Any] | None:
    p = SOURCE_DIR / f"{protocol_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_SECTION_LABELS_RU: dict[str, str] = {
    "classification": "Классификация",
    "diagnostics": "Диагностика и обследования",
    "treatment": "Лечение",
    "prevention": "Профилактика и наблюдение",
    "routing": "Госпитализация и маршрут",
    "criteria": "Критерии",
    "other": "Прочее",
}

_SECTION_ORDER: tuple[str, ...] = (
    "classification",
    "criteria",
    "diagnostics",
    "treatment",
    "prevention",
    "routing",
    "other",
)


def load_source_text_by_path(catalog_path: str) -> dict[str, Any] | None:
    """Загрузка sectioned source text по пути PDF в каталоге."""
    norm = (catalog_path or "").replace("\\", "/").strip()
    if not norm:
        return None
    doc = load_source_text(_protocol_id_from_path(norm))
    if doc and str(doc.get("path") or "").replace("\\", "/") == norm:
        return doc
    if not SOURCE_DIR.is_dir():
        return None
    for fp in SOURCE_DIR.glob("*.json"):
        try:
            row = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if str(row.get("path") or "").replace("\\", "/") == norm:
            return row
    return None


def _resolve_protocol_source_text_raw(catalog_path: str) -> dict[str, Any]:
    """Сырой source_text по разделам (без клинической фильтрации)."""
    norm = (catalog_path or "").replace("\\", "/").strip()
    doc = load_source_text_by_path(norm)
    if not doc:
        return {"available": False, "path": norm, "llm_used": False}
    sections = doc.get("sections") or {}
    toc: list[dict[str, Any]] = []
    block_count = 0
    for key in _SECTION_ORDER:
        blocks = sections.get(key) or []
        if not blocks:
            continue
        block_count += len(blocks)
        toc.append(
            {
                "id": key,
                "label": _SECTION_LABELS_RU.get(key, key),
                "count": len(blocks),
            }
        )
    return {
        "available": block_count > 0,
        "path": str(doc.get("path") or norm),
        "protocol_id": doc.get("protocol_id"),
        "title": doc.get("title") or "",
        "sections": sections,
        "toc": toc,
        "block_count": block_count,
        "section_labels": dict(_SECTION_LABELS_RU),
        "llm_used": False,
    }


def _doc_from_rich_chunks(chunks: list[dict[str, Any]], *, path: str) -> dict[str, Any] | None:
    """Синтетический doc из rich-чанков: сохраняет chunk_type и сущности для view."""
    if not chunks:
        return None
    blocks: list[dict[str, Any]] = []
    title = ""
    protocol_id = ""
    for ch in chunks:
        text = str(ch.get("text") or "").strip()
        if not text:
            continue
        title = title or str(ch.get("protocol_title_normalized") or ch.get("protocol_title") or "")
        protocol_id = protocol_id or str(ch.get("doc_id") or "")
        blocks.append(
            {
                "chunk_type": ch.get("chunk_type") or ch.get("kind"),
                "section_title": ch.get("section_title") or ch.get("title"),
                "text": text,
                "page_from": ch.get("page_from") or ch.get("page"),
                "page_to": ch.get("page_to"),
                "drugs": ch.get("drugs") or [],
                "imaging": ch.get("imaging") or [],
                "lab_tests": ch.get("lab_tests") or [],
                "procedures": ch.get("procedures") or [],
            }
        )
    if not blocks:
        return None
    return {
        "path": path,
        "protocol_id": protocol_id,
        "title": title,
        "sections": {"rich": blocks},
    }


def resolve_protocol_source_text(
    catalog_path: str,
    *,
    clinical_view: bool = True,
    rich_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Полный текст протокола + компактное клиническое представление.

    Если переданы rich_chunks (из RAG-индекса), клинический view строится на
    них - у чанков есть настоящий chunk_type и списки сущностей. Иначе
    используются пред-собранные блоки source_text с диска.
    """
    base = _resolve_protocol_source_text_raw(catalog_path)
    if not base.get("available") and not rich_chunks:
        return base
    if not clinical_view:
        return base
    from clinical_knowledge.protocol_source_view import prepare_protocol_source_view

    view = None
    rich_doc = _doc_from_rich_chunks(rich_chunks or [], path=base.get("path") or catalog_path)
    if rich_doc is not None:
        view = prepare_protocol_source_view(rich_doc)
        if not (view.get("toc") or []):
            view = None
        elif not base.get("available"):
            base.update(
                {
                    "available": True,
                    "path": rich_doc["path"],
                    "protocol_id": base.get("protocol_id") or rich_doc.get("protocol_id"),
                    "title": base.get("title") or rich_doc.get("title") or "",
                }
            )
    if view is None:
        if not base.get("available"):
            return base
        view = prepare_protocol_source_view(base)
    base["view"] = view
    base["view_available"] = bool((view.get("toc") or []))
    base["view_source"] = "rich_chunks" if rich_doc is not None and view is not None else "source_text"
    return base


def section_text_blob(doc: dict[str, Any], section_keys: list[str], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    sections = doc.get("sections") or {}
    for key in section_keys:
        for block in sections.get(key) or []:
            if not isinstance(block, dict):
                continue
            title = block.get("section_title") or key
            page = block.get("page_from")
            head = f"[{key}] {title}"
            if page:
                head += f" (стр. {page})"
            parts.append(head + "\n" + str(block.get("text") or ""))
    blob = "\n\n".join(parts)
    return blob[:max_chars]
