"""Ссылки на PDF протоколов (безопасный путь → URL API)."""
from __future__ import annotations

import re
from urllib.parse import quote, unquote

_BLOCKED_RUBRICS = frozenset({
    "output", "data", "tests", "scripts", "clients_consult", "e2e", "__pycache__",
})

_RUBRIC_RU: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология",
    "novoobrazovaniya": "Новообразования",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "pediatriya": "Педиатрия",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
}

_RE_FILE_NOISE = re.compile(
    r"^(?:КП[\s_]*\d*[._\s-]*|кп[\s_]*\d*[._\s-]*)|"
    r"(?:_оф\.?\s*опубл\.?|_офиц\.?|оф\.?\s*опубл\.?).*$",
    re.I,
)
_RE_DATE_SPACED = re.compile(r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b")


def beautify_protocol_title(raw: str | None) -> str:
    """Читаемое название протокола: без подчёркиваний из имён PDF, даты и пост. МЗ."""
    if not raw:
        return ""
    name = str(raw).strip()
    if not name:
        return ""
    if "/" in name or "\\" in name:
        name = name.replace("\\", "/").split("/")[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    def _fmt_date(m: re.Match[str]) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{d:02d}.{mo:02d}.{y}"
        return m.group(0)

    name = _RE_DATE_SPACED.sub(_fmt_date, name)
    name = re.sub(r"пост\.?\s*МЗ", "пост. МЗ", name, flags=re.I)
    name = re.sub(r"\bот\s+(\d{2}\.\d{2}\.\d{4})\b", r"от \1", name, flags=re.I)
    name = re.sub(r"№\s*(\d+)", r"№\1", name)
    name = _RE_FILE_NOISE.sub("", name).strip(" .-")
    name = re.sub(r"\(\s+", "(", name)
    name = re.sub(r"\s+\)", ")", name)
    name = re.sub(r"\s+", " ", name).strip()
    from clinical_knowledge.protocol_audience import expand_protocol_title_abbreviations

    return expand_protocol_title_abbreviations(name)


_RE_GENERIC_DIAG_TX = re.compile(
    r"(?is)^клинический\s+протокол\s*«?\s*диагностика\s+и\s+лечение\b"
)


_RE_BOILERPLATE_TITLE = re.compile(
    r"(?is)(?:устанавливает\s+общие\s+требования|"
    r"об\s+утверждении\s+клинических\s+протоколов|"
    r"заменить\s+словами|"  # обрывок правки постановления, не название КП
    r"^клинический\s+протокол\s*$|"
    r"^клинических\s+протоколов\s*$)"
)


def title_looks_truncated(title: str | None) -> bool:
    """OCR/первая строка карточки без полного названия заболевания."""
    text = str(title or "").strip()
    if not text:
        return True
    if text.count("«") != text.count("»"):
        return True
    if text.endswith(("«", "(", ",", ";", "-", "-", " - ")):
        return True
    if _RE_BOILERPLATE_TITLE.search(text):
        return True
    if _RE_GENERIC_DIAG_TX.match(text) and ("пациент" in text.lower() or "»" not in text):
        # Общий заголовок без нозологии или обрезок.
        if len(text) < 90 or "»" not in text:
            return True
    return False


def protocol_display_name(
    local_path: str | None,
    fallback: str = "",
    *,
    registry_title: str | None = None,
    prefer_filename_if_truncated: bool = False,
) -> str:
    """Читаемое название протокола для ссылки в UI."""
    file_pretty = ""
    if local_path:
        name = str(local_path).replace("\\", "/").split("/")[-1]
        file_pretty = beautify_protocol_title(name)
    reg_pretty = beautify_protocol_title(registry_title) if registry_title else ""
    fb_pretty = beautify_protocol_title(fallback) if fallback else ""

    if prefer_filename_if_truncated:
        if file_pretty and (
            title_looks_truncated(reg_pretty)
            or title_looks_truncated(registry_title)
            or not reg_pretty
        ):
            return file_pretty
        if reg_pretty and not title_looks_truncated(reg_pretty):
            return reg_pretty
        if file_pretty:
            return file_pretty
        if fb_pretty:
            return fb_pretty
        return "Протокол"

    if registry_title and str(registry_title).strip():
        t = reg_pretty
        if len(t) >= 6 and t.lower() not in ("протокол", "protocol"):
            if prefer_filename_if_truncated and title_looks_truncated(t) and file_pretty:
                return file_pretty
            return t
    if file_pretty and len(file_pretty) >= 6:
        return file_pretty
    if len(fb_pretty) >= 6:
        return fb_pretty
    return fb_pretty or "Протокол"


def normalize_protocol_path(local_path: str | None) -> str | None:
    """Нормализует путь к PDF: decode, слэши, префикс minzdrav_protocols/."""
    if not local_path:
        return None
    p = str(local_path).strip().replace("\\", "/")
    if "%" in p:
        try:
            p = unquote(p)
        except Exception:
            pass
    p = re.sub(r"/{2,}", "/", p.lstrip("/"))
    if ".." in p:
        return None
    low = p.lower()
    if not low.endswith(".pdf"):
        return None
    if not low.startswith("minzdrav_protocols/"):
        if "/" in p:
            p = f"minzdrav_protocols/{p.lstrip('/')}"
        else:
            return None
    parts = p.split("/")
    if len(parts) >= 2 and parts[1].lower() in _BLOCKED_RUBRICS:
        return None
    return p


def protocol_rubric_slug(local_path: str | None) -> str | None:
    norm = normalize_protocol_path(local_path)
    if not norm:
        return None
    parts = norm.split("/")
    if len(parts) >= 3:
        return parts[1]
    return None


def protocol_rubric_label(local_path: str | None) -> str | None:
    slug = protocol_rubric_slug(local_path)
    if not slug:
        return None
    return _RUBRIC_RU.get(slug, slug.replace("-", " ").replace("_", " "))


def protocol_pdf_api_path(local_path: str | None) -> str | None:
    """Возвращает относительный URL `/api/protocol-pdf?path=…` или None."""
    p = normalize_protocol_path(local_path)
    if not p:
        return None
    return f"/api/protocol-pdf?path={quote(p, safe='')}"


def protocol_nav_api_path(
    local_path: str | None,
    *,
    section: str | None = None,
    q: str | None = None,
) -> str | None:
    """Относительный URL навигатора протокола `/proto-viewer.html?path=…`."""
    p = normalize_protocol_path(local_path)
    if not p:
        return None
    url = f"/proto-viewer.html?path={quote(p, safe='')}"
    if section and str(section).strip():
        url += f"&section={quote(str(section).strip(), safe='')}"
    if q and str(q).strip():
        url += f"&q={quote(str(q).strip()[:120], safe='')}"
    return url


def protocol_basename_key(local_path: str | None) -> str:
    """Ключ дедупа: имя PDF без рубрики (один КП может лежать в нескольких папках)."""
    norm = normalize_protocol_path(local_path)
    if not norm:
        return ""
    return norm.rsplit("/", 1)[-1].lower()


def protocol_title_key(title: str | None) -> str:
    t = beautify_protocol_title(title or "")
    if not t:
        return ""
    t = t.lower().replace("ё", "е")
    t = re.sub(r"\d{2}\.\d{2}\.\d{4}", " ", t)
    t = re.sub(r"№\s*\d+", " ", t)
    t = re.sub(r"пост\.?\s*мз", " ", t, flags=re.I)
    t = re.sub(r"[^a-zа-я0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedupe_protocol_rows(
    rows: list[dict] | None,
    *,
    path_key: str = "path",
    score_key: str = "confidence_score",
) -> list[dict]:
    """Убирает дубли одного КП (тот же файл / то же название) - оставляет лучший score."""
    if not rows:
        return []

    def _score(row: dict) -> float:
        try:
            return float(row.get(score_key) or row.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    by_path: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = normalize_protocol_path(str(row.get(path_key) or ""))
        if not path:
            continue
        entry = dict(row)
        entry[path_key] = path
        prev = by_path.get(path)
        if prev is None or _score(entry) > _score(prev):
            by_path[path] = entry

    by_base: dict[str, dict] = {}
    for row in by_path.values():
        path = str(row.get(path_key) or "")
        bk = protocol_basename_key(path) or f"path:{path}"
        prev = by_base.get(bk)
        if prev is None or _score(row) > _score(prev):
            by_base[bk] = row
        elif prev is not None:
            dups = list(prev.get("duplicate_catalog_paths") or [])
            other = str(row.get(path_key) or "")
            if other and other not in dups and other != prev.get(path_key):
                dups.append(other)
            if dups:
                prev = dict(prev)
                prev["duplicate_catalog_paths"] = dups
                by_base[bk] = prev

    by_title: dict[str, dict] = {}
    for row in by_base.values():
        tk = protocol_title_key(str(row.get("title") or row.get("registry_title") or ""))
        if not tk:
            tk = f"path:{row.get(path_key) or ''}"
        prev = by_title.get(tk)
        if prev is None or _score(row) > _score(prev):
            by_title[tk] = row

    out = list(by_title.values())
    out.sort(key=lambda r: (-_score(r), str(r.get(path_key) or "")))
    return out


def content_disposition_inline(filename: str) -> str:
    """Content-Disposition с поддержкой кириллицы (RFC 5987)."""
    safe = (filename or "protocol.pdf").replace('"', "")
    ascii_name = re.sub(r"[^\x20-\x7E]", "_", safe) or "protocol.pdf"
    utf8_name = quote(safe, safe="")
    return f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{utf8_name}'


def protocol_link_payload(
    local_path: str | None,
    *,
    title: str | None = None,
    matched_icd_codes: list[str] | None = None,
    section: str | None = None,
    pages: str | None = None,
    icd_verified: bool = False,
    q: str | None = None,
) -> dict | None:
    """Единый объект ссылки для API/UI.

    Основная ссылка - навигатор протокола (`nav_url` / `url`).
    `pdf_url` остаётся для точечного открытия PDF (страница, скачивание).
    """
    norm = normalize_protocol_path(local_path)
    if not norm:
        return None
    pdf_url = protocol_pdf_api_path(norm)
    nav_url = protocol_nav_api_path(norm, section=section, q=q)
    if not pdf_url or not nav_url:
        return None
    display = protocol_display_name(norm, registry_title=title)
    rubric = protocol_rubric_label(norm)
    out: dict = {
        "path": norm,
        "nav_url": nav_url,
        "url": nav_url,
        "pdf_url": pdf_url,
        "title": display,
        "rubric": rubric,
    }
    if title and title.strip() and title.strip() != display:
        out["registry_title"] = title.strip()
    if matched_icd_codes:
        out["matched_icd_codes"] = list(matched_icd_codes)
    if section:
        out["section"] = section
    if pages:
        out["pages"] = pages
    if icd_verified:
        out["icd_verified"] = True
    return out
