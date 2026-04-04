"""Паспорт документа и разбиение на логические протоколы внутри одного PDF."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import SPECIALTY_FROM_FOLDER
from .entities_extract import (
    extract_care_settings,
    extract_icd10,
    extract_populations,
    keywords_from_text,
)

RE_ACT_DATE = re.compile(
    r"(?:(\d{1,2})[.\s]+([а-яё]+)\s+(\d{4})\s*г?)|((\d{4})-(\d{2})-(\d{2}))",
    re.I,
)
RE_NUMBER = re.compile(r"№\s*([\d\-–—/]+)", re.I)
RE_POSTANOVLENIE = re.compile(
    r"постановлени[ея]\s+([^.]+\.)",
    re.I,
)
SPLIT_PROTOCOL = re.compile(
    r"(?=(?:УТВЕРЖДЕНО|Утверждено|КЛИНИЧЕСКИЙ\s+ПРОТОКОЛ|Клинический\s+протокол))",
    re.I,
)


def _folder_from_path(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 else ""


def _title_from_filename(name: str) -> str:
    return re.sub(r"\.pdf$", "", name, flags=re.I).replace("_", " ").strip()


def split_logical_documents(full_norm: str) -> list[tuple[str, int]]:
    """Возвращает список (фрагмент_текста, start_offset в full_norm)."""
    t = full_norm.strip()
    if not t:
        return [("", 0)]
    parts = list(SPLIT_PROTOCOL.finditer(t))
    if len(parts) <= 1:
        return [(t, 0)]
    idxs = sorted({0, *[m.start() for m in parts], len(t)})
    out: list[tuple[str, int]] = []
    for i in range(len(idxs) - 1):
        a, b = idxs[i], idxs[i + 1]
        chunk = t[a:b].strip()
        if chunk:
            out.append((chunk, a))
    return out or [(t, 0)]


def build_act_block(text_head: str) -> dict[str, Any]:
    """Реквизиты нормативного акта (эвристика по шапке документа)."""
    head = (text_head or "")[:8000]
    low = head.lower()
    issuing = "Министерство здравоохранения Республики Беларусь"
    if "министерств" in low and "здравоохранен" in low:
        pass
    else:
        issuing = ""

    act_type = "постановление"
    if "приказ" in low[:2000]:
        act_type = "приказ"

    date_s = ""
    m = RE_ACT_DATE.search(head)
    if m:
        date_s = m.group(0)[:80]

    num = ""
    mn = RE_NUMBER.search(head)
    if mn:
        num = mn.group(1).strip()

    amendments: list[str] = []
    repeals: list[str] = []
    for line in head.split("\n"):
        l = line.strip()
        if re.search(r"изменени|дополнени|редакци", l, re.I):
            if len(l) < 300:
                amendments.append(l)
        if re.search(r"утратил|отмен|замен", l, re.I):
            if len(l) < 300:
                repeals.append(l)

    status = "действует"
    if re.search(r"утратил\s+силу", head, re.I):
        status = "утратил силу"

    return {
        "issuing_body": issuing or None,
        "act_type": act_type,
        "date": date_s or None,
        "number": num or None,
        "official_publication": None,
        "effective_date": None,
        "status": status,
        "amendments": amendments[:20],
        "repeals": repeals[:20],
    }


def build_protocol_passport(
    logical_text: str,
    file_title: str,
    category_slug: str,
) -> dict[str, Any]:
    t = logical_text[:120_000]
    icd = extract_icd10(t)
    pops = extract_populations(t)
    care = extract_care_settings(t)

    # Название протокола из первых строк
    proto_title = file_title
    for line in t.split("\n")[:40]:
        line = line.strip()
        if re.search(r"клиническ(?:ий|ого)\s+протокол", line, re.I):
            proto_title = re.sub(
                r"^УТВЕРЖДЕНО\s*", "", line, flags=re.I
            ).strip()[:500]
            break
        if len(line) > 20 and "протокол" in line.lower():
            proto_title = line[:500]
            break

    clinical_domain = SPECIALTY_FROM_FOLDER.get(category_slug, category_slug)
    topic = file_title[:200]

    return {
        "protocol_title": proto_title,
        "clinical_domain": clinical_domain,
        "topic": topic,
        "population": pops,
        "care_setting": care,
        "technology_level": [],
        "target_users": ["врачи", "медицинские организации"],
        "icd10_codes": icd,
        "key_terms": keywords_from_text(t[:15_000], 40),
    }


def build_document_json(
    doc_id: str,
    logical_id_suffix: str,
    source_path: str,
    file_name: str,
    logical_text: str,
    raw_head: str,
    page_offset_base: int,
    extraction_confidence: float,
) -> dict[str, Any]:
    category_slug = _folder_from_path(source_path)
    title_base = _title_from_filename(file_name)

    act = build_act_block(raw_head[:12_000] + "\n" + logical_text[:4000])
    passport = build_protocol_passport(logical_text, title_base, category_slug)

    full_id = f"{doc_id}_{logical_id_suffix}" if logical_id_suffix else doc_id

    return {
        "doc_id": full_id,
        "parent_pdf_doc_id": doc_id,
        "logical_index": logical_id_suffix,
        "file_name": file_name,
        "source_path": source_path,
        "title": passport["protocol_title"],
        "subtitle": title_base if passport["protocol_title"] != title_base else None,
        "act": act,
        "protocol_passport": passport,
        "page_offset_base": page_offset_base,
        "extraction_confidence": extraction_confidence,
    }
