"""Разметка ОХЛП: секции 4.1 / 4.2 / 4.3 (+ соседние) из текста инструкции."""
from __future__ import annotations

import re
from typing import Any

# Якоря Decision 88 / типовых ОХЛП (номер или русское название).
_SECTION_SPECS: list[tuple[str, re.Pattern[str]]] = [
    (
        "indications_4_1",
        re.compile(
            r"(?im)^\s*(?:4\.1\b[^\n]{0,80}|Показания(?:\s+к\s+применению)?)\s*$"
        ),
    ),
    (
        "posology_4_2",
        re.compile(
            r"(?im)^\s*(?:4\.2\b[^\n]{0,80}|Режим дозирования|"
            r"Способ применения(?:\s+и\s+дозы)?)\s*$"
        ),
    ),
    (
        "contraindications_4_3",
        re.compile(
            r"(?im)^\s*(?:4\.3\b[^\n]{0,80}|Противопоказания)\s*$"
        ),
    ),
    (
        "warnings_4_4",
        re.compile(
            r"(?im)^\s*(?:4\.4\b[^\n]{0,100}|Особые указания|"
            r"Меры предосторожности)\s*$"
        ),
    ),
    (
        "interactions_4_5",
        re.compile(
            r"(?im)^\s*(?:4\.5\b[^\n]{0,100}|Взаимодействие(?:\s+с\s+другими)?)\s*$"
        ),
    ),
]

_NEXT_MAJOR = re.compile(r"(?im)^\s*(?:4\.\d+|5\.\s*[А-ЯA-Z]|6\.\s*[А-ЯA-Z])\b")


def _normalize_lines(text: str) -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.split("\n")]
    return lines


def _find_heading(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    for i, ln in enumerate(lines):
        if pattern.match(ln):
            return i
    # fallback: heading embedded at start of longer line
    for i, ln in enumerate(lines):
        if pattern.search(ln) and len(ln) < 160:
            return i
    return None


def _body_until_next(lines: list[str], start: int) -> str:
    """Текст после заголовка до следующего 4.x / 5. / 6. заголовка."""
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if _NEXT_MAJOR.match(ln) and body:
            break
        # другой известный заголовок секции
        if any(p.match(ln) for _, p in _SECTION_SPECS) and body:
            break
        body.append(ln)
    # убрать пустые края
    while body and not body[0]:
        body.pop(0)
    while body and not body[-1]:
        body.pop()
    text = "\n".join(body).strip()
    return text


def split_oxlp_sections(text: str) -> dict[str, Any]:
    """Вырезать секции ОХЛП. Не штрафует: при провале needs_human=true."""
    lines = _normalize_lines(text)
    sections: dict[str, list[str]] = {
        "indications_4_1": [],
        "posology_4_2": [],
        "contraindications_4_3": [],
        "warnings_4_4": [],
        "interactions_4_5": [],
    }
    found: dict[str, int] = {}
    for key, pat in _SECTION_SPECS:
        idx = _find_heading(lines, pat)
        if idx is None:
            continue
        found[key] = idx
        body = _body_until_next(lines, idx)
        if body:
            # один блок текстом; список из абзацев
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            sections[key] = paras or ([body] if body else [])

    core_ok = all(sections[k] for k in ("indications_4_1", "posology_4_2", "contraindications_4_3"))
    partial = any(sections[k] for k in sections)
    needs_human = not core_ok
    return {
        "sections": sections,
        "parse": {
            "ok": core_ok,
            "method": "heading_regex_v1",
            "needs_human": needs_human,
            "found_keys": sorted(found.keys()),
            "partial": partial and not core_ok,
        },
    }


def build_label_record(
    *,
    reg_id: str,
    text: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать JSON label по схеме плана §9 (без клинических ПДн)."""
    meta = dict(meta or {})
    split = split_oxlp_sections(text)
    pdf_meta = meta.get("pdf_s") if isinstance(meta.get("pdf_s"), dict) else {}
    return {
        "reg_id": reg_id,
        "status": meta.get("status") or "active",
        "trade_name_ru": meta.get("trade_name_ru") or "",
        "inn": meta.get("inn") or "",
        "atc": meta.get("atc") or "",
        "forms": meta.get("forms") or [],
        "form_text": meta.get("form_text") or "",
        "rx_otc": meta.get("rx_otc") or "",
        "term_from": meta.get("term_from") or "",
        "term_to": meta.get("term_to") or "",
        "nd_changes": meta.get("nd_changes") or [],
        "pdf_s": {
            "url": pdf_meta.get("url") or meta.get("url_s") or "",
            "sha256": pdf_meta.get("sha256") or meta.get("pdf_s_sha256") or "",
            "bytes": pdf_meta.get("bytes") or meta.get("pdf_s_bytes") or 0,
        },
        "sections": split["sections"],
        "parse": split["parse"],
    }
