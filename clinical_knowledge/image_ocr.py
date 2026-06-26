"""OCR и распознавание фото документов (КЗ, бланки анализов)."""
from __future__ import annotations

import io
import os

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
)

_OCR_PROMPT_RU = (
    "Извлеки весь читаемый текст с изображения медицинского документа "
    "(консультативное заключение или бланк анализов, русский язык). "
    "Верни только текст документа, без комментариев и markdown. Сохрани абзацы."
)


def is_image_filename(filename: str) -> bool:
    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    if dot < 0:
        return False
    return name[dot:] in IMAGE_EXTENSIONS


def sniff_image_payload(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def _guess_image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _gemini_ocr_enabled() -> bool:
    return os.environ.get("PATIENT_OCR_GEMINI", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ocr_tesseract(data: bytes) -> tuple[str, list[str]]:
    warns: list[str] = []
    try:
        from PIL import Image
    except ImportError:
        return "", []
    try:
        import pytesseract
    except ImportError:
        return "", []
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        lang = os.environ.get("PATIENT_OCR_LANG", "rus+eng")
        txt = pytesseract.image_to_string(img, lang=lang)
        if not (txt or "").strip():
            return "", ["Tesseract не распознал текст на фото"]
        return (txt or "").strip(), ["Текст извлечён из фото через Tesseract OCR"]
    except Exception as e:
        msg = str(e).lower()
        if "tesseract" in msg or "not installed" in msg or "no such file" in msg:
            return "", []
        return "", [f"Tesseract OCR: {e!s}"]


def _ocr_gemini_vision(data: bytes) -> tuple[str, list[str]]:
    if not _gemini_ocr_enabled():
        return "", []
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return "", []
    try:
        from rag_server import _extract_gemini_text, get_gemini

        model = get_gemini()
        mime = _guess_image_mime(data)
        resp = model.generate_content(
            [_OCR_PROMPT_RU, {"mime_type": mime, "data": data}],
            generation_config={
                "max_output_tokens": int(os.environ.get("PATIENT_OCR_GEMINI_MAX_TOKENS", "8192")),
                "temperature": 0,
            },
        )
        txt = _extract_gemini_text(resp)
        if not txt.strip():
            return "", ["Gemini не вернул текст с фото"]
        return txt.strip(), ["Текст извлечён из фото через Gemini Vision"]
    except Exception as e:
        return "", [f"Gemini Vision OCR: {e!s}"]


def ocr_image_bytes(data: bytes, filename: str = "") -> tuple[str, list[str]]:
    """Tesseract (если установлен) → Gemini Vision (если есть API key)."""
    txt, warns = _ocr_tesseract(data)
    if txt.strip():
        return txt, warns

    gtxt, gwarns = _ocr_gemini_vision(data)
    if gtxt.strip():
        return gtxt, gwarns

    combined = [w for w in warns + gwarns if w]
    if not combined:
        combined = [
            "Не удалось распознать фото. Загрузите PDF из клиники или переснимите при хорошем свете."
        ]
    return "", combined
