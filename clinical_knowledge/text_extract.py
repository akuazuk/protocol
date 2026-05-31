"""Извлечение текста из файлов КЗ для CLI/batch (PDF, DOCX, TXT, JSON)."""
from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt", ".md", ".json"})


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
    if ext == ".pdf" or data[:5].startswith(b"%PDF-"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    if ext == ".docx" or (data[:2] == b"PK" and ext != ".pdf"):
        return extract_docx_text(data)
    return data.decode("utf-8", errors="replace")


def extract_docx_text(data: bytes) -> str:
    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
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
