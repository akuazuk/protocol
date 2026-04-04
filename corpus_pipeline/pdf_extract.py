"""Извлечение текста из PDF постранично; смещения для маппинга чанков на страницы; опциональный OCR."""
from __future__ import annotations

import bisect
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from .text_normalize import normalize_text

MIN_NATIVE = int(os.environ.get("CORPUS_MIN_CHARS_PAGE", "80"))


@dataclass
class PageBlock:
    page_no: int  # 1-based
    raw_text: str
    normalized_text: str
    extraction_confidence: float
    ocr_used: bool


@dataclass
class ExtractedDocument:
    source_path: str
    doc_id: str
    pages: list[PageBlock]
    full_raw: str
    full_normalized: str
    """Начало каждой страницы в full_normalized (0-based индекс страницы → смещение)."""
    page_starts: list[int]
    warnings: list[str] = field(default_factory=list)


def _doc_id_from_path(rel: str) -> str:
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24]


def _ocr_page_image(page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pix = page.get_pixmap(dpi=150)
        mode = "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, lang="rus+eng")
    except Exception:
        return ""


def extract_pdf(pdf_path: Path, rel_path: str) -> ExtractedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise SystemExit("Установите: pip install pymupdf") from e

    doc = fitz.open(pdf_path)
    warnings: list[str] = []
    pages: list[PageBlock] = []
    use_ocr = os.environ.get("CORPUS_USE_OCR", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    for i in range(len(doc)):
        page = doc[i]
        raw = (page.get_text("text") or "").strip()
        ocr_used = False
        conf = 0.92 if len(raw) >= MIN_NATIVE else 0.45

        if len(raw) < MIN_NATIVE and use_ocr:
            ocr_raw = _ocr_page_image(page)
            if len(ocr_raw.strip()) > len(raw):
                raw = ocr_raw.strip()
                ocr_used = True
                conf = 0.55
                if not raw:
                    warnings.append(f"page_{i + 1}: пусто после OCR")
        elif len(raw) < MIN_NATIVE:
            warnings.append(
                f"page_{i + 1}: мало текста; CORPUS_USE_OCR=1 + pytesseract для OCR"
            )

        norm = normalize_text(raw)
        pages.append(
            PageBlock(
                page_no=i + 1,
                raw_text=raw,
                normalized_text=norm,
                extraction_confidence=conf,
                ocr_used=ocr_used,
            )
        )

    doc.close()

    joiner = "\n\n"
    full_normalized = joiner.join(p.normalized_text for p in pages)
    full_raw = joiner.join(p.raw_text for p in pages)

    page_starts: list[int] = []
    cur = 0
    for i, p in enumerate(pages):
        page_starts.append(cur)
        cur += len(p.normalized_text)
        if i < len(pages) - 1:
            cur += len(joiner)

    return ExtractedDocument(
        source_path=rel_path.replace("\\", "/"),
        doc_id=_doc_id_from_path(rel_path.replace("\\", "/")),
        pages=pages,
        full_raw=full_raw,
        full_normalized=full_normalized,
        page_starts=page_starts,
        warnings=warnings,
    )


def span_to_pages(page_starts: list[int], full_len: int, start: int, end: int) -> tuple[int, int]:
    """Интервал [start, end) в full_normalized → (page_from, page_to), 1-based."""
    if not page_starts:
        return 1, 1
    start = max(0, min(start, full_len))
    end = max(start, min(end, full_len))
    npg = len(page_starts)

    def page_num_for(offset: int) -> int:
        j = max(0, bisect.bisect_right(page_starts, offset) - 1)
        j = min(j, npg - 1)
        return j + 1

    if end <= start:
        p = page_num_for(start)
        return p, p
    pf = page_num_for(start)
    pt = page_num_for(max(start, end - 1))
    return min(pf, pt), max(pf, pt)
