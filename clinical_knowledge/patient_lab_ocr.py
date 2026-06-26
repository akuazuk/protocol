"""OCR для фото бланков анализов (B2C)."""
from __future__ import annotations

import io
import os
from typing import Any


def _ocr_image_bytes(data: bytes) -> tuple[str, list[str]]:
    warns: list[str] = []
    try:
        from PIL import Image
    except ImportError:
        return "", ["PIL не установлен — OCR недоступен"]
    try:
        import pytesseract
    except ImportError:
        return "", ["pytesseract не установлен — установите для OCR фото"]
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        lang = os.environ.get("PATIENT_OCR_LANG", "rus+eng")
        txt = pytesseract.image_to_string(img, lang=lang)
        if not (txt or "").strip():
            warns.append("OCR не распознал текст на изображении")
        else:
            warns.append("Текст извлечён через OCR")
        return (txt or "").strip(), warns
    except Exception as e:
        return "", [f"OCR ошибка: {e!s}"]


def _is_image_filename(filename: str) -> bool:
    low = (filename or "").lower()
    return low.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"))


def extract_lab_text_from_bytes(data: bytes, filename: str = "") -> tuple[str, list[str]]:
    """PDF/DOCX через стандартный extractor; для фото — OCR при малом тексте."""
    from rag_server import extract_consult_text_from_bytes

    txt, warns = extract_consult_text_from_bytes(data, filename)
    if len((txt or "").strip()) >= 40:
        return txt, warns
    use_ocr = os.environ.get("PATIENT_LAB_OCR", "1").strip().lower() in ("1", "true", "yes", "on")
    if not use_ocr:
        return txt, warns
    if not _is_image_filename(filename):
        low = (filename or "").lower()
        if not low.endswith(".pdf"):
            return txt, warns
    ocr_txt, ocr_warns = _ocr_image_bytes(data)
    if len(ocr_txt) > len((txt or "").strip()):
        return ocr_txt, warns + ocr_warns
    return txt, warns
