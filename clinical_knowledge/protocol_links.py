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


def protocol_display_name(
    local_path: str | None,
    fallback: str = "",
    *,
    registry_title: str | None = None,
) -> str:
    """Читаемое название протокола для ссылки в UI."""
    if registry_title and str(registry_title).strip():
        t = beautify_protocol_title(registry_title)
        if len(t) >= 6 and t.lower() not in ("протокол", "protocol"):
            return t
    if local_path:
        name = str(local_path).replace("\\", "/").split("/")[-1]
        pretty = beautify_protocol_title(name)
        if len(pretty) >= 6:
            return pretty
    fb = beautify_protocol_title(fallback) if fallback else ""
    if len(fb) >= 6:
        return fb
    return fb or "Протокол"


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
) -> dict | None:
    """Единый объект ссылки для API/UI."""
    norm = normalize_protocol_path(local_path)
    if not norm:
        return None
    url = protocol_pdf_api_path(norm)
    if not url:
        return None
    display = protocol_display_name(norm, registry_title=title)
    rubric = protocol_rubric_label(norm)
    out: dict = {
        "path": norm,
        "pdf_url": url,
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
