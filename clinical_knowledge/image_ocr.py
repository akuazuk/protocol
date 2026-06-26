"""OCR и распознавание фото документов (КЗ, бланки анализов)."""
from __future__ import annotations

import io
import os

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
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


def ocr_image_bytes(data: bytes) -> tuple[str, list[str]]:
    warns: list[str] = []
    try:
        from PIL import Image
    except ImportError:
        return "", [
            "Для фото установите Pillow (pip install Pillow). "
            "Или загрузите PDF из клиники."
        ]
    try:
        import pytesseract
    except ImportError:
        return "", [
            "Для фото установите pytesseract и Tesseract OCR. "
            "Или загрузите PDF из клиники."
        ]
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        lang = os.environ.get("PATIENT_OCR_LANG", "rus+eng")
        txt = pytesseract.image_to_string(img, lang=lang)
        if not (txt or "").strip():
            warns.append("OCR не распознал текст — переснимите при хорошем свете или загрузите PDF.")
        else:
            warns.append("Текст извлечён из фото через OCR")
        return (txt or "").strip(), warns
    except Exception as e:
        return "", [f"OCR ошибка: {e!s}"]
