"""Паспорт нозологии из Ilex: точное имя КП, МКБ главы 1, статус акта.

HTML Ilex в git не храним. В рантайме читаем компактный JSONL.
Выключить: ILEX_PASSPORTS=0.
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PASSPORTS = ROOT / "output" / "registry" / "ilex_protocol_passports.jsonl"

_ICD_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,4})?)\b")
_TAG_RE = re.compile(r"<[^>]+>")
_OGL_RE = re.compile(
    r'<ogl-tag[^>]*tag-value="([^"]+)"[^>]*>',
    re.IGNORECASE,
)
_NAME_DATE_N = re.compile(
    r"от\s+(\d{2})\.(\d{2})\.(\d{4})\s+N\s+(\d+)",
    re.IGNORECASE,
)
_PROTO_IN_NAME = re.compile(
    r'Клиническ(?:им|ий|ого)\s+протокол(?:ом|а|ы)?\s+"([^"]+)"',
    re.IGNORECASE,
)
_SHIFR_RE = re.compile(
    r"шифр по международн.{0,220}пересмотра:\s*(.+?)(?:\.\s+\d+\.|Глава\s+2|$)",
    re.IGNORECASE | re.DOTALL,
)
_GLAVA1_RE = re.compile(
    r"Глава\s+1[\.:]?(.*?)(?:Глава\s+2[\.:]|$)",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_STOP = frozenset(
    """
    диагностика лечение пациентов пациент население взрослое детское
    клинический протокол заболеваний заболевание оказание медицинской
    помощи медицинская и с со при после
    """.split()
)
_PLACEHOLDER_YEAR = 2090


def passports_enabled() -> bool:
    raw = (os.environ.get("ILEX_PASSPORTS") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def passports_path() -> Path:
    env = (os.environ.get("ILEX_PASSPORTS_PATH") or "").strip()
    return Path(env) if env else DEFAULT_PASSPORTS


def _ms_to_iso(value: Any) -> str | None:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    seconds = ms / 1000.0 if ms > 10_000_000_000 else float(ms)
    try:
        day = datetime.fromtimestamp(seconds, tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None
    if day.year >= _PLACEHOLDER_YEAR:
        return None
    return day.isoformat()


def parse_approval_from_name(name: str) -> tuple[str | None, str | None]:
    match = _NAME_DATE_N.search(name or "")
    if not match:
        return None, None
    day, month, year, number = match.groups()
    return f"{year}-{month}-{day}", str(int(number))


def infer_audience(title: str) -> str:
    low = (title or "").lower()
    child = "детск" in low
    adult = "взросл" in low
    if child and adult:
        return "any"
    if child:
        return "child"
    if adult:
        return "adult"
    return "any"


def norm_status(raw: str | None) -> str:
    text = (raw or "").lower()
    if "тратил" in text or "отменен" in text or "отменён" in text:
        return "repealed"
    if "действ" in text:
        return "active"
    return "unknown"


def strip_markup(raw: str) -> str:
    text = (raw or "").replace("&nbsp;", " ").replace("&#xa0;", " ").replace("\xa0", " ")
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _uniq(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _icd_from_text(text: str, *, limit: int = 16) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _ICD_RE.finditer((text or "").upper()):
        code = match.group(1)
        if code[0] in "YWVT" or code in seen:
            continue
        seen.add(code)
        out.append(code)
        if len(out) >= limit:
            break
    return out


def extract_ogl_values(html_text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for match in _OGL_RE.finditer(html_text or ""):
        raw = html_lib.unescape(match.group(1)).strip()
        if not raw:
            continue
        level_m = re.search(r'tag-level="(\d+)"', match.group(0), re.I)
        level = int(level_m.group(1)) if level_m else 1
        rows.append((level, raw))
    return rows


def extract_protocol_titles(name: str, html_text: str, meta_inner: list[str] | None) -> list[str]:
    titles = list(meta_inner or [])
    titles.extend(_PROTO_IN_NAME.findall(name or ""))
    for level, value in extract_ogl_values(html_text):
        if level != 1:
            continue
        low = value.lower()
        if "клинический протокол" not in low:
            continue
        inner = value
        quoted = re.search(r"[«\"]([^»\"]+)[»\"]", value)
        if quoted:
            inner = quoted.group(1).strip()
        titles.append(inner)
    cleaned = []
    for title in titles:
        text = strip_markup(title).strip(" «»\"")
        if len(text) < 12:
            continue
        if text.lower().startswith("постановление"):
            continue
        cleaned.append(text)
    return _uniq(cleaned)


def extract_chapter_titles(html_text: str) -> list[str]:
    out: list[str] = []
    for level, value in extract_ogl_values(html_text):
        if level >= 2 and value.lower().startswith("глава"):
            out.append(strip_markup(value))
    return _uniq(out)[:24]


def extract_icd_chapter1(html_text: str) -> list[str]:
    plain = strip_markup(html_text or "")
    shifr = _SHIFR_RE.search(plain)
    if shifr:
        codes = _icd_from_text(shifr.group(1), limit=12)
        if codes:
            return codes
    glava = _GLAVA1_RE.search(plain)
    if glava:
        codes = _icd_from_text(glava.group(1)[:3500], limit=12)
        if codes:
            return codes
    return _icd_from_text(plain[:1800], limit=8)


def _ogl_inner_title(value: str) -> str | None:
    low = (value or "").lower()
    if "клинический протокол" not in low:
        return None
    quoted = re.search(r"[«\"]([^»\"]+)[»\"]", value)
    inner = quoted.group(1).strip() if quoted else value
    text = strip_markup(inner).strip(" «»\"")
    if len(text) < 12 or text.lower().startswith("постановление"):
        return None
    return text


def split_protocol_html_sections(html_text: str) -> list[tuple[str, str]]:
    """Куски HTML по каждому клиническому протоколу внутри акта."""
    if not html_text:
        return []
    starts: list[tuple[int, str]] = []
    for match in _OGL_RE.finditer(html_text):
        level_m = re.search(r'tag-level="(\d+)"', match.group(0), re.I)
        level = int(level_m.group(1)) if level_m else 1
        if level != 1:
            continue
        title = _ogl_inner_title(html_lib.unescape(match.group(1)))
        if not title:
            continue
        starts.append((match.start(), title))
    if not starts:
        return []
    sections: list[tuple[str, str]] = []
    for idx, (pos, title) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(html_text)
        sections.append((title, html_text[pos:end]))
    return sections


def parse_ilex_document(
    *,
    meta: dict[str, Any],
    html_text: str = "",
    card: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Один акт Ilex → одна или несколько строк паспорта (по внутреннему КП)."""
    card = card if isinstance(card, dict) else {}
    name = str(meta.get("name") or card.get("name") or "")
    approval_date, approval_number = parse_approval_from_name(name)
    if not approval_date:
        approval_date = _ms_to_iso(meta.get("date") or card.get("date"))
    status_raw = meta.get("status")
    if not status_raw and isinstance(card.get("status"), dict):
        status_raw = card["status"].get("description")
    valid_from = _ms_to_iso(meta.get("takeEffectDate") or card.get("takeEffectDate"))
    valid_to = _ms_to_iso(meta.get("terminateEffectDate") or card.get("terminateEffectDate"))
    titles = extract_protocol_titles(name, html_text, list(meta.get("inner_protocols") or []))
    if not titles:
        titles = [name[:240]] if name else []
    section_list = split_protocol_html_sections(html_text)
    sections = {title.casefold(): chunk for title, chunk in section_list}
    fallback_icd = extract_icd_chapter1(html_text) if html_text and len(titles) == 1 else []
    chapters = extract_chapter_titles(html_text)
    bank = str(meta.get("bank") or "").strip()
    doc_id = meta.get("doc_id")
    ilex_id = str(meta.get("ilex_id") or (f"{bank}/{doc_id}" if bank and doc_id else "")).strip()
    rows: list[dict[str, Any]] = []
    for title in titles:
        chunk = sections.get(title.casefold()) or ""
        if not chunk:
            for key, html_chunk in sections.items():
                if title.casefold() in key or key in title.casefold():
                    chunk = html_chunk
                    break
        icd = extract_icd_chapter1(chunk) if chunk else list(fallback_icd)
        rows.append(
            {
                "ilex_id": ilex_id,
                "bank": bank,
                "doc_id": doc_id,
                "act_title": name[:300],
                "protocol_title": title,
                "audience": infer_audience(title),
                "status": norm_status(str(status_raw or "")),
                "approval_date": approval_date,
                "approval_number": approval_number,
                "approval_year": (approval_date or "")[:4] or None,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "icd10_primary": list(icd),
                "chapters": list(chapters),
            }
        )
    return rows


def load_html_prefix(folder: Path, *, pages: int | None = None) -> str:
    pdir = folder / "pages"
    if not pdir.is_dir():
        return ""
    chunks: list[str] = []
    files = sorted(
        pdir.glob("*.html"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else 999,
    )
    selected = files if pages is None else files[:pages]
    for path in selected:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def build_passports_from_dump(dump_dir: Path) -> list[dict[str, Any]]:
    docs = dump_dir / "docs"
    if not docs.is_dir():
        raise FileNotFoundError(f"нет каталога docs в {dump_dir}")
    rows: list[dict[str, Any]] = []
    for folder in sorted(p for p in docs.iterdir() if p.is_dir()):
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        card: dict[str, Any] = {}
        card_path = folder / "card.json"
        if card_path.is_file():
            try:
                loaded = json.loads(card_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    card = loaded
            except json.JSONDecodeError:
                card = {}
        html_text = load_html_prefix(folder)
        rows.extend(parse_ilex_document(meta=meta, html_text=html_text, card=card))
    return rows


def write_passports_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


@lru_cache(maxsize=1)
def load_ilex_passports() -> tuple[dict[str, Any], ...]:
    if not passports_enabled():
        return tuple()
    return tuple(_read_jsonl(passports_path()))


def clear_ilex_passport_cache() -> None:
    load_ilex_passports.cache_clear()


def _approval_key(year: str | None, number: str | None) -> tuple[str, str] | None:
    num = str(number or "").strip()
    year_s = str(year or "").strip()[:4]
    if not num or not year_s:
        return None
    try:
        num = str(int(num))
    except ValueError:
        pass
    return year_s, num


def _card_approval_key(card: dict[str, Any]) -> tuple[str, str] | None:
    approval = card.get("approval") if isinstance(card.get("approval"), dict) else {}
    date = str(approval.get("date") or card.get("valid_from") or "")
    number = approval.get("number")
    return _approval_key(date[:4], str(number) if number is not None else "")


def _card_search_text(card: dict[str, Any]) -> str:
    path = str(card.get("source_path") or "").replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(
        [
            str(card.get("title") or ""),
            str(card.get("condition_label") or ""),
            path,
        ]
    )


def _passport_fits_card(row: dict[str, Any], card_tokens: set[str]) -> bool:
    title_tokens = _title_tokens(str(row.get("protocol_title") or ""))
    hit = title_tokens & card_tokens
    if not hit:
        return False
    # один общий корень мало: «лечение» уже в стопе, но «заболеван» бывает часто
    strong = {tok for tok in hit if len(tok) >= 7}
    return bool(strong) or len(hit) >= 2


def _fold_token(word: str) -> str:
    if word.startswith("гипертон") or word.startswith("гипертенз"):
        return "гипертенз"
    return word


def _title_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for word in re.findall(r"[а-яёa-z0-9]{5,}", (text or "").lower()):
        if word in _TITLE_STOP:
            continue
        folded = _fold_token(word)
        found.add(folded)
        if len(folded) >= 8:
            found.add(folded[:6])
    return found


def _nosology_overlap(query: str, title: str) -> float:
    q = _title_tokens(query)
    t = _title_tokens(title)
    if not q or not t:
        return 0.0
    hit = q & t
    if not hit:
        return 0.0
    return len(hit) / max(3.0, min(len(q), 8) * 0.5)


def overlay_ilex_passports(cards: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Обогатить карты точным именем / МКБ главы 1 / статусом Ilex. Карты мутируются."""
    if not isinstance(cards, list) or not passports_enabled():
        return cards or []
    passports = load_ilex_passports()
    if not passports:
        return cards
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pdfs_by_key: dict[tuple[str, str], set[str]] = {}
    for row in passports:
        key = _approval_key(row.get("approval_year"), row.get("approval_number"))
        if not key:
            continue
        by_key.setdefault(key, []).append(row)
    for card in cards:
        key = _card_approval_key(card)
        if not key:
            continue
        fname = str(card.get("source_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if fname:
            pdfs_by_key.setdefault(key, set()).add(fname)
    from clinical_knowledge.protocol_links import title_looks_truncated

    try:
        from clinical_knowledge.kp_validity import looks_omnibus
    except Exception:  # noqa: BLE001
        looks_omnibus = lambda _card: False  # noqa: E731

    for card in cards:
        key = _card_approval_key(card)
        if not key:
            continue
        candidates = by_key.get(key) or []
        if not candidates:
            continue
        card_tokens = _title_tokens(_card_search_text(card))
        matched = [row for row in candidates if _passport_fits_card(row, card_tokens)]
        if not matched:
            titles = _uniq([str(row.get("protocol_title") or "") for row in candidates])
            ilex_ids = {str(row.get("ilex_id") or "") for row in candidates if row.get("ilex_id")}
            # Один акт, один КП, один PDF: имя файла может быть «АГ», без нозологии.
            if (
                len(titles) == 1
                and len(ilex_ids) <= 1
                and len(pdfs_by_key.get(key) or ()) == 1
            ):
                fname = str(card.get("source_path") or "")
                year, num = key
                if year in fname and num in fname.replace("№", "").replace(" ", ""):
                    matched = list(candidates)
        if not matched:
            continue
        titles = _uniq([str(row.get("protocol_title") or "") for row in matched])
        icd: list[str] = []
        for row in matched:
            icd.extend(str(code) for code in (row.get("icd10_primary") or []) if code)
        icd = _uniq(icd)[:16]
        in_force = any(row.get("status") == "active" for row in matched)
        repealed = all(row.get("status") == "repealed" for row in matched)
        valid_to = next((row.get("valid_to") for row in matched if row.get("valid_to")), None)
        ilex_id = str(matched[0].get("ilex_id") or "")
        card["ilex_id"] = ilex_id
        card["ilex_protocol_titles"] = titles
        if titles:
            existing = str(card.get("condition_label") or "").strip()
            joined = " | ".join(titles)
            if not existing:
                card["condition_label"] = joined
            elif joined.lower() not in existing.lower():
                card["condition_label"] = f"{existing} | {joined}"
        if len(titles) == 1 and title_looks_truncated(str(card.get("title") or "")):
            card["title"] = titles[0]
            card["ilex_title_overlay"] = True
        if icd:
            old_primary = [str(x) for x in (card.get("icd10_primary") or []) if x]
            if not old_primary:
                card["icd10_primary"] = icd[:12]
                card["ilex_icd_overlay"] = "fill"
            elif looks_omnibus(card) and 1 <= len(icd) <= 12:
                mentions = [str(x) for x in (card.get("icd10_mentions") or []) if x]
                for code in old_primary:
                    if code not in icd and code not in mentions:
                        mentions.append(code)
                card["icd10_mentions"] = mentions[:80]
                card["icd10_primary"] = icd[:12]
                card["ilex_icd_overlay"] = "omnibus"
        if repealed and not in_force:
            card["status"] = "repealed"
            if valid_to and not card.get("valid_to"):
                card["valid_to"] = valid_to
    return cards


def match_passports_for_diagnosis(
    diag_text: str,
    *,
    icd_codes: list[str] | None = None,
    audience: str | None = None,
    visit_year: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Прямой подбор нозологии по названию паспорта Ilex (без PDF-карт)."""
    try:
        from clinical_knowledge.dx_query_expand import expand_diagnosis_query

        query = expand_diagnosis_query(diag_text or "") or (diag_text or "").strip()
    except Exception:  # noqa: BLE001
        query = (diag_text or "").strip()
    codes = [str(c).strip().upper() for c in (icd_codes or []) if c]
    roots = {c[:3] for c in codes if len(c) >= 3}
    aud = (audience or "").strip().lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in load_ilex_passports():
        if row.get("status") == "repealed":
            continue
        title = str(row.get("protocol_title") or "")
        score = _nosology_overlap(query, title)
        icd = [str(x).upper() for x in (row.get("icd10_primary") or []) if x]
        icd_part = 0.0
        if codes and icd:
            if any(code in icd for code in codes):
                icd_part = 0.9
            elif {_c[:3] for _c in icd if len(_c) >= 3} & roots:
                icd_part = 0.55
        if score < 0.35 and icd_part < 0.55:
            continue
        pop = str(row.get("audience") or "any")
        if aud in {"adult", "child"} and pop in {"adult", "child"} and pop != aud:
            continue
        year = str(row.get("approval_year") or "")
        recency = 0.0
        if year.isdigit():
            recency = min(1.0, max(0.0, (int(year) - 2015) / 12.0))
        raw = 100.0 * (0.7 * min(1.0, score) + 0.25 * icd_part + 0.05 * recency)
        if visit_year and year and year > visit_year:
            # ещё не действует в год визита - не отбрасываем без дат, только чуть ниже
            raw *= 0.92
        scored.append((round(raw, 2), row))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("protocol_title") or "")))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, row in scored:
        key = str(row.get("protocol_title") or "").casefold()
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["match_score"] = score
        out.append(item)
        if len(out) >= limit:
            break
    return out


def enrich_diagnosis_with_ilex(
    diag_text: str,
    *,
    audience: str | None = None,
    icd_codes: list[str] | None = None,
) -> str:
    """Добавить точные названия КП Ilex к запросу, если нозология уже узнана."""
    base = (diag_text or "").strip()
    if not base or not passports_enabled():
        return base
    try:
        hits = match_passports_for_diagnosis(
            base,
            icd_codes=icd_codes,
            audience=audience,
            limit=1,
        )
    except Exception:  # noqa: BLE001
        return base
    extra = [
        str(row.get("protocol_title") or "").strip()
        for row in hits
        if float(row.get("match_score") or 0) >= 40
    ]
    if not extra:
        return base
    return re.sub(r"\s+", " ", (base + " " + " ".join(extra)).strip())
