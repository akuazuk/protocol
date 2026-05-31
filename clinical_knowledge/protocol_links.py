"""Ссылки на PDF протоколов (безопасный путь → URL API)."""
from __future__ import annotations

from urllib.parse import quote


def protocol_pdf_api_path(local_path: str | None) -> str | None:
    """Возвращает относительный URL `/api/protocol-pdf?path=…` или None."""
    if not local_path:
        return None
    p = str(local_path).strip().replace("\\", "/").lstrip("/")
    if not p or ".." in p:
        return None
    if not p.lower().startswith("minzdrav_protocols/"):
        return None
    if not p.lower().endswith(".pdf"):
        return None
    return f"/api/protocol-pdf?path={quote(p, safe='')}"


def protocol_display_name(local_path: str | None, fallback: str = "") -> str:
    if not local_path:
        return fallback or "протокол"
    name = str(local_path).replace("\\", "/").split("/")[-1]
    return name or fallback or "протокол"
