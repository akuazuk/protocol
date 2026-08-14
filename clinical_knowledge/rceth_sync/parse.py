"""Парсинг HTML выдачи / карточки Refbank (без сети)."""
from __future__ import annotations

import re
from html import unescape
from typing import Any

_DETAIL_RE = re.compile(
    r"/Refbank/reestr_lekarstvennih_sredstv/details/([A-Za-z0-9_]+)",
    re.I,
)
_PDF_RE = re.compile(r"/NDfiles/instr/([A-Za-z0-9_]+)(_[ps])\.pdf", re.I)
_COUNT_RE = re.compile(r"(\d+)\s+записей\s*-\s*(\d+)\s+страниц", re.I)
_ROW_RE = re.compile(r"<tr([^>]*)>([\s\S]*?)</tr>", re.I)
_CLASS_STATUS = re.compile(r"\b(unterm|annul|pause|my-discountinued|my-notactual)\b", re.I)


def parse_result_counts(html: str) -> tuple[int | None, int | None]:
    m = _COUNT_RE.search(html or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _pdfs_for_id(html_chunk: str, reg_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _PDF_RE.finditer(html_chunk or ""):
        rid, kind = m.group(1), m.group(2).lower()
        if rid != reg_id:
            continue
        url = f"/NDfiles/instr/{rid}{kind}.pdf"
        if kind == "_s":
            out["url_s"] = url
        elif kind == "_p":
            out["url_p"] = url
    return out


_INSTR_NOISE = re.compile(
    r"\s*(?:инструкция\s+(?:специалиста|пациента)\s*)+",
    re.I,
)


def _clean_cell(text: str) -> str:
    t = _INSTR_NOISE.sub(" ", text or "")
    return re.sub(r"\s+", " ", t).strip()


def parse_search_results(html: str) -> list[dict[str, Any]]:
    """Строки из results HTML. Статус - из class tr; PDF - из ячейки."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tr_m in _ROW_RE.finditer(html or ""):
        attrs, body = tr_m.group(1), tr_m.group(2)
        ids = _DETAIL_RE.findall(body)
        if not ids:
            continue
        reg_id = ids[0]
        if reg_id in seen:
            continue
        seen.add(reg_id)
        flags = sorted({c.lower() for c in _CLASS_STATUS.findall(attrs + " " + body)})
        status = "active"
        if "unterm" in flags:
            status = "expired"
        elif "annul" in flags:
            status = "annulled"
        elif "pause" in flags:
            status = "paused"
        pdfs = _pdfs_for_id(body, reg_id)
        # грубый текст ячеек для trade/inn (best-effort)
        cells = [
            unescape(re.sub(r"<[^>]+>", " ", c)).strip()
            for c in re.findall(r"<td[^>]*>([\s\S]*?)</td>", body, re.I)
        ]
        cells = [_clean_cell(c) for c in cells if c.strip()]
        trade = cells[1] if len(cells) > 1 else ""
        inn = cells[2] if len(cells) > 2 else ""
        form = cells[3] if len(cells) > 3 else ""
        rows.append(
            {
                "reg_id": reg_id,
                "status": status,
                "status_flags": flags,
                "trade_name_ru": trade[:240],
                "inn": inn[:120],
                "form_text": form[:240],
                "url_detail": f"/Refbank/reestr_lekarstvennih_sredstv/details/{reg_id}",
                "url_s": pdfs.get("url_s") or "",
                "url_p": pdfs.get("url_p") or "",
                "has_s_pdf": bool(pdfs.get("url_s")),
                "has_p_pdf": bool(pdfs.get("url_p")),
            }
        )
    return rows


def parse_detail_card(html: str, reg_id: str = "") -> dict[str, Any]:
    """Паспорт с карточки details (без разделов ОХЛП - они в PDF)."""
    text = re.sub(r"<script[\s\S]*?</script>", "", html or "", flags=re.I)
    plain = unescape(re.sub(r"<[^>]+>", "\n", text))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in plain.splitlines() if ln.strip()]

    def after(label: str) -> str:
        for i, ln in enumerate(lines):
            if ln.rstrip(":").lower() == label.lower() or ln.lower().startswith(label.lower() + ":"):
                if ":" in ln:
                    rest = ln.split(":", 1)[1].strip()
                    if rest:
                        return rest[:240]
                if i + 1 < len(lines):
                    return lines[i + 1][:240]
        return ""

    if not reg_id:
        m = _DETAIL_RE.search(html or "")
        # fallback from pdf links
        pm = _PDF_RE.search(html or "")
        reg_id = pm.group(1) if pm else ""

    pdfs = {}
    for m in _PDF_RE.finditer(html or ""):
        rid, kind = m.group(1), m.group(2).lower()
        if reg_id and rid != reg_id:
            continue
        if not reg_id:
            reg_id = rid
        url = f"/NDfiles/instr/{rid}{kind}.pdf"
        if kind == "_s":
            pdfs["url_s"] = url
        else:
            pdfs["url_p"] = url

    atc = after("Код АТХ")
    composition = after("Состав лекарственного средства")
    rx = after("Порядок отпуска")
    nd_changes: list[str] = []
    for i, ln in enumerate(lines):
        if "Изменение в нормативной" in ln:
            for j in range(i + 1, min(i + 8, len(lines))):
                if lines[j].startswith("Номер разрешения") or lines[j].startswith("Субстанц"):
                    break
                if lines[j].startswith("изменение") or "ОХЛП" in lines[j] or "пр. №" in lines[j]:
                    nd_changes.append(lines[j][:300])
            break

    return {
        "reg_id": reg_id,
        "inn": composition,
        "atc": atc,
        "rx_otc": rx,
        "url_s": pdfs.get("url_s") or "",
        "url_p": pdfs.get("url_p") or "",
        "has_s_pdf": bool(pdfs.get("url_s")),
        "has_p_pdf": bool(pdfs.get("url_p")),
        "nd_changes": nd_changes,
        "url_detail": f"/Refbank/reestr_lekarstvennih_sredstv/details/{reg_id}" if reg_id else "",
    }


def merge_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Дедуп по reg_id; active предпочитаем при коллизии."""
    by_id: dict[str, dict[str, Any]] = {}
    rank = {"active": 0, "paused": 1, "expired": 2, "annulled": 3}
    for row in rows:
        rid = str(row.get("reg_id") or "")
        if not rid:
            continue
        prev = by_id.get(rid)
        if not prev or rank.get(str(row.get("status")), 9) < rank.get(str(prev.get("status")), 9):
            by_id[rid] = dict(row)
        else:
            # дополнить пустые поля
            for k, v in row.items():
                if v and not prev.get(k):
                    prev[k] = v
    return [by_id[k] for k in sorted(by_id)]
