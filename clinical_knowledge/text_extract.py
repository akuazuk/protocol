"""Извлечение текста из файлов КЗ для CLI/batch (PDF, DOCX, TXT, JSON)."""
from __future__ import annotations

import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt", ".md", ".json"})


def strip_file_prefix(data: bytes) -> bytes:
    """Убрать BOM/пробелы перед ZIP или %PDF-."""
    return data.lstrip(b"\x00 \t\r\n\xef\xbb\xbf\xfe\xff")


def is_zip_payload(data: bytes) -> bool:
    return strip_file_prefix(data)[:2] == b"PK"


def normalize_pdf_bytes(data: bytes, *, max_scan: int = 8192) -> bytes | None:
    """Найти маркер %PDF- (BOM, пробелы, служебный префикс перед PDF)."""
    if not data:
        return None
    stripped = strip_file_prefix(data)
    if stripped[:5].startswith(b"%PDF-"):
        return stripped
    idx = data.find(b"%PDF-", 0, max_scan)
    if idx >= 0:
        return data[idx:]
    return None


def extract_pdf_text_pypdf(data: bytes, *, max_pages: int = 200) -> tuple[str, list[str], str | None]:
    """pypdf; error_code: encrypted | unreadable | None."""
    warnings: list[str] = []
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", warnings, "unreadable"
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        warnings.append(f"pypdf: не читается как PDF ({e!s})")
        return "", warnings, "unreadable"
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            return "", warnings, "encrypted"
    parts: list[str] = []
    pages = list(reader.pages or [])
    if max_pages > 0 and len(pages) > max_pages:
        warnings.append(
            f"Обработаны только первые {max_pages} стр. из {len(pages)} (лимит страниц)."
        )
        pages = pages[:max_pages]
    for i, page in enumerate(pages):
        try:
            t = (page.extract_text() or "").strip()
        except Exception as e:
            warnings.append(f"Стр. {i + 1}: pypdf ({e!s})")
            t = ""
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip(), warnings, None


def extract_pdf_text_pymupdf(data: bytes, *, max_pages: int = 200) -> tuple[str, list[str]]:
    """PyMuPDF (fitz) - fallback для PDF МИС, где pypdf часто пустой."""
    warnings: list[str] = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "", warnings
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        warnings.append(f"PyMuPDF: {e!s}")
        return "", warnings
    parts: list[str] = []
    n_pages = doc.page_count
    limit = n_pages if max_pages <= 0 else min(n_pages, max_pages)
    if max_pages > 0 and n_pages > max_pages:
        warnings.append(
            f"Обработаны только первые {max_pages} стр. из {n_pages} (лимит страниц)."
        )
    for i in range(limit):
        try:
            t = (doc.load_page(i).get_text() or "").strip()
        except Exception as e:
            warnings.append(f"Стр. {i + 1}: PyMuPDF ({e!s})")
            t = ""
        if t:
            parts.append(t)
    try:
        doc.close()
    except Exception:
        pass
    return "\n\n".join(parts).strip(), warnings


def _pdf_ocr_enabled() -> bool:
    return (os.environ.get("CONSULT_PDF_OCR") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def extract_pdf_text_ocr(
    data: bytes,
    *,
    max_pages: int = 6,
) -> tuple[str, list[str]]:
    """OCR fallback для PDF без текстового слоя: рендер страниц -> OCR изображения."""
    warnings: list[str] = []
    if max_pages <= 0:
        return "", warnings
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "", warnings
    try:
        from clinical_knowledge.image_ocr import ocr_image_bytes
    except Exception:
        return "", warnings

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        warnings.append(f"PDF OCR: PyMuPDF ({e!s})")
        return "", warnings

    parts: list[str] = []
    n_pages = doc.page_count
    limit = min(n_pages, max_pages)
    if n_pages > max_pages:
        warnings.append(f"PDF OCR: обработаны только первые {max_pages} стр. из {n_pages}.")

    for i in range(limit):
        try:
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            png = pix.tobytes("png")
            txt, _ = ocr_image_bytes(png, f"pdf_page_{i + 1}.png")
            if txt.strip():
                parts.append(txt.strip())
        except Exception as e:
            warnings.append(f"PDF OCR: стр. {i + 1} ({e!s})")
    try:
        doc.close()
    except Exception:
        pass

    full = "\n\n".join(parts).strip()
    if full:
        warnings.append("текст извлечён через OCR для сканированного PDF.")
    return full, warnings


def extract_pdf_text_bytes(
    data: bytes,
    *,
    max_pages: int = 200,
) -> tuple[str, list[str], str | None]:
    """PDF: pypdf, затем PyMuPDF. error_code: encrypted | unreadable | None."""
    pdf_data = normalize_pdf_bytes(data) or data
    warnings: list[str] = []
    if normalize_pdf_bytes(data) is not None and pdf_data is not data:
        warnings.append("пропущен служебный префикс до маркера %PDF-.")

    txt, w, err = extract_pdf_text_pypdf(pdf_data, max_pages=max_pages)
    warnings.extend(w)
    if err == "encrypted":
        return "", warnings, "encrypted"
    if txt.strip():
        return txt, warnings, None

    txt2, w2 = extract_pdf_text_pymupdf(pdf_data, max_pages=max_pages)
    warnings.extend(w2)
    if txt2.strip():
        warnings.append("текст извлечён через PyMuPDF (pypdf вернул пусто).")
        return txt2, warnings, None

    if _pdf_ocr_enabled():
        ocr_pages = int(os.environ.get("CONSULT_PDF_OCR_MAX_PAGES", "6") or "6")
        txt3, w3 = extract_pdf_text_ocr(pdf_data, max_pages=max(1, ocr_pages))
        warnings.extend(w3)
        if txt3.strip():
            return txt3, warnings, None

    if err == "unreadable":
        return "", warnings, "unreadable"
    return "", warnings, None


def extract_text_from_bytes(data: bytes, *, suffix: str = "") -> str:
    ext = suffix.lower()
    if ext == ".json":
        try:
            obj = json.loads(data.decode("utf-8", errors="replace"))
            if isinstance(obj, dict):
                for key in ("raw_text", "text", "content", "body"):
                    if obj.get(key):
                        return str(obj[key])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            return data.decode("utf-8", errors="replace")
    if ext == ".pdf" or normalize_pdf_bytes(data) is not None:
        txt, _, err = extract_pdf_text_bytes(data)
        if txt.strip():
            return txt
        if is_zip_payload(data):
            return extract_docx_text(data)
        if err:
            return ""
    if ext == ".docx" or is_zip_payload(data):
        return extract_docx_text(data)
    return data.decode("utf-8", errors="replace")


def extract_docx_text(data: bytes) -> str:
    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    payload = strip_file_prefix(data)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
    except (zipfile.BadZipFile, ET.ParseError, KeyError):
        return ""
    paras: list[str] = []
    for p in root.iter(f"{w_ns}p"):
        parts = [(n.text or "") for n in p.iter(f"{w_ns}t")]
        line = "".join(parts).strip()
        if line:
            paras.append(line)
    return "\n\n".join(paras).strip()


def extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return extract_text_from_bytes(path.read_bytes(), suffix=suffix)
    return extract_text_from_bytes(path.read_bytes(), suffix=suffix)
