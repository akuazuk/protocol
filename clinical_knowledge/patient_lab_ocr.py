"""OCR для фото бланков анализов (B2C)."""
from __future__ import annotations

import os

from .image_ocr import is_image_filename, ocr_image_bytes, sniff_image_payload


def extract_lab_text_from_bytes(data: bytes, filename: str = "") -> tuple[str, list[str]]:
    """PDF/DOCX через стандартный extractor; для фото - OCR."""
    from rag_server import extract_consult_text_from_bytes

    if is_image_filename(filename) or sniff_image_payload(data):
        return ocr_image_bytes(data)

    txt, warns = extract_consult_text_from_bytes(data, filename)
    if len((txt or "").strip()) >= 40:
        return txt, warns
    use_ocr = os.environ.get("PATIENT_LAB_OCR", "1").strip().lower() in ("1", "true", "yes", "on")
    if not use_ocr:
        return txt, warns
    ocr_txt, ocr_warns = ocr_image_bytes(data)
    if len(ocr_txt) > len((txt or "").strip()):
        return ocr_txt, warns + ocr_warns
    return txt, warns
