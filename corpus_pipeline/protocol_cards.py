"""Единый реестр карточек клинических протоколов (retrieval + rule matching)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import OUT_REGISTRY, OUTPUT_ROOT, SPECIALTY_FROM_FOLDER
from .entities_extract import extract_icd10

RE_APPROVAL_NUM = re.compile(r"(?:№|_|\s)(\d{1,4})(?:\.pdf|$|\s)", re.I)
RE_APPROVAL_DATE = re.compile(
    r"(?:от\s+)?(\d{1,2})[.\s]+(\d{1,2})[.\s]+(\d{4})|(\d{4})[-_](\d{2})[-_](\d{2})|(\d{4})[_\s](\d{1,3})(?:\.pdf|$|\s)",
    re.I,
)
RE_POST_MZ = re.compile(r"пост[.\s_]*(?:МЗ|мз)", re.I)

POPULATION_ADULT_MARKERS = (
    "взросл",
    "взр",
    "вз. нас",
    "вз_нас",
    "вз.нас",
)
POPULATION_CHILD_MARKERS = (
    "детск",
    "дет. нас",
    "дет_нас",
    "дет.нас",
    "дет ",
    "детс",
    "ребён",
    "ребен",
    "новорожд",
    "перинат",
)


def _slug_part(text: str, max_len: int = 48) -> str:
    t = (text or "").lower()
    t = re.sub(r"\.pdf$", "", t, flags=re.I)
    t = re.sub(r"[^a-z0-9а-яё]+", "_", t, flags=re.I)
    t = re.sub(r"_+", "_", t).strip("_")
    if len(t) > max_len:
        t = t[:max_len].rstrip("_")
    return t or "protocol"


def infer_population_from_text(*parts: str) -> str:
    """adult | child | any — по имени файла и заголовку."""
    blob = " ".join(p for p in parts if p).lower()
    is_child = any(m in blob for m in POPULATION_CHILD_MARKERS)
    is_adult = any(m in blob for m in POPULATION_ADULT_MARKERS)
    if is_child and not is_adult:
        return "child"
    if is_adult and not is_child:
        return "adult"
    if is_child and is_adult:
        return "any"
    return "any"


def infer_approval_from_filename(filename: str) -> dict[str, str | None]:
    """Номер и дата постановления из имени PDF."""
    fn = filename or ""
    num = None
    m_num = re.search(r"№\s*([\d\-–—/]+)", fn, re.I)
    if m_num:
        num = m_num.group(1).strip()
    else:
        m_alt = re.search(r"пост[_\s]*(?:МЗ|мз)[_\s]*(?:\d{4})[_\s]*(\d{1,4})", fn, re.I)
        if m_alt:
            num = m_alt.group(1).strip()
        else:
            m_tail = re.search(r"_(\d{2,4})\.pdf$", fn, re.I)
            if m_tail:
                num = m_tail.group(1)

    date_iso = None
    m_d = re.search(r"(\d{1,2})[.\s]+(\d{1,2})[.\s]+(\d{4})", fn)
    if m_d:
        d, mo, y = m_d.group(1), m_d.group(2), m_d.group(3)
        date_iso = f"{y}-{int(mo):02d}-{int(d):02d}"
    else:
        m_y = re.search(r"(?:пост[_\s]*(?:МЗ|мз)[_\s]*)?(\d{4})[_\s](\d{1,3})", fn, re.I)
        if m_y:
            date_iso = f"{m_y.group(1)}-01-01"

    act_type = "postanovlenie" if RE_POST_MZ.search(fn) else "unknown"
    return {
        "act_type": act_type,
        "approval_date": date_iso,
        "approval_number": num,
    }


def _care_setting_codes(raw: list[str]) -> list[str]:
    mapping = {
        "амбулатор": "ambulatory",
        "стационар": "inpatient",
        "скорой": "emergency",
        "неотложн": "urgent",
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        low = (item or "").lower()
        for needle, code in mapping.items():
            if needle in low and code not in seen:
                seen.add(code)
                out.append(code)
    return out


def build_protocol_id(
    specialty_slug: str,
    source_path: str,
    logical_index: str,
    approval_number: str | None,
    population: str,
) -> str:
    base = Path(source_path).stem
    slug = _slug_part(base, 56)
    parts = [_slug_part(specialty_slug, 24), slug]
    if population and population != "any":
        parts.append(population)
    if approval_number:
        parts.append(str(approval_number).replace("/", "-"))
    if logical_index:
        parts.append(logical_index.lower())
    return "_".join(p for p in parts if p)[:120]


def card_from_document(doc: dict[str, Any], manifest_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Собрать карточку из output/documents/*.json."""
    source_path = str(doc.get("source_path") or "")
    file_name = str(doc.get("file_name") or Path(source_path).name)
    parts = source_path.replace("\\", "/").split("/")
    specialty_slug = parts[1] if len(parts) > 1 else ""
    specialty_ru = SPECIALTY_FROM_FOLDER.get(specialty_slug, specialty_slug)

    passport = doc.get("protocol_passport") or {}
    act = doc.get("act") or {}
    fn_meta = infer_approval_from_filename(file_name)

    title = str(doc.get("title") or passport.get("protocol_title") or file_name)
    population = infer_population_from_text(file_name, title, " ".join(passport.get("population") or []))

    icd_all = list(dict.fromkeys(
        (passport.get("icd10_codes") or [])
        + extract_icd10(file_name)
        + extract_icd10(source_path)
        + extract_icd10(title)
        + extract_icd10(str(doc.get("subtitle") or ""))
        + extract_icd10((doc.get("text") or {}).get("normalized", "")[:80_000])
    ))[:80]
    icd_primary = icd_all[:12]

    approval_number = act.get("number") or fn_meta.get("approval_number")
    approval_date = fn_meta.get("approval_date")
    if act.get("date") and not approval_date:
        m = re.search(r"(\d{4})", str(act.get("date")))
        if m:
            approval_date = f"{m.group(1)}-01-01"

    logical_index = str(doc.get("logical_index") or "")
    doc_id = str(doc.get("doc_id") or "")
    pdf_doc_id = str(doc.get("pdf_doc_id") or doc.get("parent_pdf_doc_id") or "")

    status = "active"
    if (act.get("status") or "").lower().find("утрат") >= 0:
        status = "repealed"

    protocol_id = build_protocol_id(
        specialty_slug, source_path, logical_index, approval_number, population
    )

    replaces: list[str] = []
    for r in act.get("repeals") or []:
        if isinstance(r, str) and r.strip():
            replaces.append(r.strip()[:240])

    return {
        "protocol_id": protocol_id,
        "doc_id": doc_id,
        "pdf_doc_id": pdf_doc_id,
        "logical_index": logical_index or None,
        "source_path": source_path,
        "source_url": (manifest_row or {}).get("url"),
        "sha256": (manifest_row or {}).get("sha256"),
        "specialty_slug": specialty_slug,
        "specialty_ru": specialty_ru,
        "title": title[:500],
        "population": population,
        "care_setting": _care_setting_codes(passport.get("care_setting") or []),
        "approval": {
            "type": act.get("act_type") or fn_meta.get("act_type") or "postanovlenie",
            "issuing_body": act.get("issuing_body"),
            "date": approval_date,
            "number": approval_number,
            "valid_from": act.get("effective_date"),
        },
        "replaces_text": replaces[:5],
        "icd10_primary": icd_primary,
        "icd10_all": icd_all,
        "status": status,
        "extraction_confidence": doc.get("extraction_confidence"),
    }


def card_from_protocols_row(
    row: dict[str, Any],
    manifest_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Минимальная карточка только из protocols.json (без полного текста)."""
    source_path = str(row.get("path") or "")
    file_name = str(row.get("filename") or Path(source_path).name)
    parts = source_path.replace("\\", "/").split("/")
    specialty_slug = str(row.get("category") or (parts[1] if len(parts) > 1 else ""))
    specialty_ru = SPECIALTY_FROM_FOLDER.get(specialty_slug, specialty_slug)
    title = str(row.get("title") or file_name)
    fn_meta = infer_approval_from_filename(file_name)
    population = infer_population_from_text(file_name, title)
    icd = extract_icd10(file_name + " " + title)

    protocol_id = build_protocol_id(
        specialty_slug,
        source_path,
        "",
        fn_meta.get("approval_number"),
        population,
    )

    return {
        "protocol_id": protocol_id,
        "doc_id": None,
        "pdf_doc_id": None,
        "logical_index": None,
        "source_path": source_path,
        "source_url": (manifest_row or {}).get("url"),
        "sha256": (manifest_row or {}).get("sha256"),
        "specialty_slug": specialty_slug,
        "specialty_ru": specialty_ru,
        "title": title[:500],
        "population": population,
        "care_setting": [],
        "approval": {
            "type": fn_meta.get("act_type") or "postanovlenie",
            "issuing_body": "Министерство здравоохранения Республики Беларусь",
            "date": fn_meta.get("approval_date"),
            "number": fn_meta.get("approval_number"),
            "valid_from": None,
        },
        "replaces_text": [],
        "icd10_primary": icd[:12],
        "icd10_all": icd,
        "status": "active",
        "extraction_confidence": None,
    }


def load_manifest_by_path(manifest_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not manifest_path.is_file():
        return out
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel = str(row.get("relative_path") or "").replace("\\", "/")
        if rel:
            out[rel] = row
    return out


def build_all_protocol_cards(
    root: Path,
    *,
    documents_dir: Path | None = None,
    protocols_json: Path | None = None,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Собрать карточки: приоритет — documents/*.json, иначе protocols.json."""
    root = Path(root)
    documents_dir = documents_dir or (root / "output" / "documents")
    protocols_json = protocols_json or (root / "protocols.json")
    manifest_path = manifest_path or (root / "minzdrav_protocols" / "_manifest.jsonl")
    manifest = load_manifest_by_path(manifest_path)

    cards: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    if documents_dir.is_dir():
        for p in sorted(documents_dir.glob("*.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = str(doc.get("source_path") or "").replace("\\", "/")
            card = card_from_document(doc, manifest.get(rel))
            cards.append(card)
            if card.get("doc_id"):
                seen_doc_ids.add(str(card["doc_id"]))

    if protocols_json.is_file():
        try:
            rows = json.loads(protocols_json.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        covered_paths = {c.get("source_path") for c in cards}
        for row in rows:
            rel = str(row.get("path") or "").replace("\\", "/")
            if rel in covered_paths:
                continue
            cards.append(card_from_protocols_row(row, manifest.get(rel)))

    cards.sort(key=lambda c: (c.get("specialty_slug") or "", c.get("title") or ""))
    return cards


def write_protocol_cards_jsonl(cards: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")


def default_protocol_cards_path() -> Path:
    return OUT_REGISTRY / "protocol_cards.jsonl"
