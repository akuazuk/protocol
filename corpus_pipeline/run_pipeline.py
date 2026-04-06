#!/usr/bin/env python3
"""
Полный прогон: PDF → документы, чанки, таблицы, сущности, реестр.

  cd Protocol && python3 -m corpus_pipeline.run_pipeline

Переменные: CORPUS_PDF_ROOT, CORPUS_OUTPUT_ROOT, CORPUS_USE_OCR — см. output/README.md
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus_pipeline.chunk_build import (
    build_chunks_for_section,
    build_table_chunks_for_document,
)
from corpus_pipeline.config import (
    OUT_CHUNKS,
    OUT_DOCS,
    OUT_ENTITIES,
    OUT_REGISTRY,
    OUT_TABLES,
    OUTPUT_ROOT,
    PDF_ROOT,
)
from corpus_pipeline.entities_extract import extract_icd10
from corpus_pipeline.passport_build import build_document_json, split_logical_documents
from corpus_pipeline.pdf_extract import extract_pdf
from corpus_pipeline.section_detect import detect_sections
from corpus_pipeline.tables_extract import extract_tables_from_pdf, merge_multipage_tables


def _ensure_dirs() -> None:
    for d in (OUT_DOCS, OUT_CHUNKS, OUT_TABLES, OUT_ENTITIES, OUT_REGISTRY):
        d.mkdir(parents=True, exist_ok=True)


def _aggregate_entities(all_chunks: list[dict], all_tables: list[dict]) -> dict:
    icd: dict[str, int] = defaultdict(int)
    pops: dict[str, int] = defaultdict(int)
    care: dict[str, int] = defaultdict(int)
    drugs: dict[str, int] = defaultdict(int)
    terms: dict[str, int] = defaultdict(int)
    procedures: dict[str, int] = defaultdict(int)

    for ch in all_chunks:
        for c in ch.get("icd10_codes") or []:
            icd[c] += 1
        for p in ch.get("population") or []:
            pops[p] += 1
        for c in ch.get("care_setting") or []:
            care[c] += 1
        for d in ch.get("drugs") or []:
            drugs[str(d)[:120]] += 1
        for k in ch.get("keywords") or []:
            terms[k] += 1

    for t in all_tables:
        for row in t.get("rows") or []:
            for cell in row:
                for code in extract_icd10(str(cell)):
                    icd[code] += 1

    return {
        "icd10_codes": dict(sorted(icd.items(), key=lambda x: -x[1])[:500]),
        "populations": dict(sorted(pops.items(), key=lambda x: -x[1])[:200]),
        "care_settings": dict(sorted(care.items(), key=lambda x: -x[1])[:100]),
        "procedures": dict(sorted(procedures.items(), key=lambda x: -x[1])[:200]),
        "drugs": dict(sorted(drugs.items(), key=lambda x: -x[1])[:500]),
        "terms": dict(sorted(terms.items(), key=lambda x: -x[1])[:1000]),
    }


def main() -> None:
    if not PDF_ROOT.is_dir():
        raise SystemExit(
            f"Нет каталога с PDF: {PDF_ROOT}\n"
            "Укажите CORPUS_PDF_ROOT или положите файлы в minzdrav_protocols/"
        )

    _ensure_dirs()
    for p in OUT_DOCS.glob("*.json"):
        p.unlink()

    pdfs = sorted(PDF_ROOT.rglob("*.pdf"))
    all_chunks_flat: list[dict] = []
    all_tables_flat: list[dict] = []
    registry_rows: list[dict] = []

    for pdf_path in pdfs:
        rel = str(pdf_path.relative_to(ROOT)).replace("\\", "/")
        file_name = pdf_path.name

        try:
            extracted = extract_pdf(pdf_path, rel)
        except Exception as e:
            print(f"SKIP {rel}: {e}", file=sys.stderr)
            continue

        full = extracted.full_normalized
        tables_merged = merge_multipage_tables(extract_tables_from_pdf(pdf_path))
        for ti, tb in enumerate(tables_merged):
            all_tables_flat.append(
                {
                    "table_id": f"{extracted.doc_id}_tbl_{ti}",
                    "pdf_doc_id": extracted.doc_id,
                    "source_path": rel,
                    "file_name": file_name,
                    **tb,
                }
            )

        logical_parts = split_logical_documents(full)

        for li, (logical_text, logical_off) in enumerate(logical_parts):
            suffix = f"L{li}" if len(logical_parts) > 1 else ""

            avg_conf = (
                sum(p.extraction_confidence for p in extracted.pages)
                / max(1, len(extracted.pages))
            )
            raw_head = "\n".join(p.raw_text for p in extracted.pages[:3])

            doc = build_document_json(
                extracted.doc_id,
                suffix,
                rel,
                file_name,
                logical_text,
                raw_head,
                logical_off,
                avg_conf,
            )
            full_doc_id = doc["doc_id"]

            sections = detect_sections(full_doc_id, logical_text)
            for sec in sections:
                sec["start_char"] = sec["start_char"] + logical_off
                sec["end_char"] = sec["end_char"] + logical_off
                sec["text"] = full[sec["start_char"] : sec["end_char"]]

            doc_chunks: list[dict] = []
            for sec in sections:
                doc_chunks.extend(
                    build_chunks_for_section(
                        full_doc_id,
                        full,
                        extracted.page_starts,
                        sec,
                        sec.get("section_type") or "body",
                    )
                )
            # Таблицы pdfplumber — отдельные чанки table_block (только первый логический документ PDF)
            if li == 0:
                doc_chunks.extend(
                    build_table_chunks_for_document(
                        full_doc_id,
                        full,
                        extracted.page_starts,
                        tables_merged,
                        rel,
                        file_name,
                    )
                )

            for ch in doc_chunks:
                ch["source_path"] = rel
                ch["file_name"] = file_name

            doc["chunk_count"] = len(doc_chunks)
            doc["table_count"] = len(tables_merged)
            doc["page_count"] = len(extracted.pages)
            doc["pdf_doc_id"] = extracted.doc_id
            doc["text"] = {
                "normalized": logical_text,
                "pdf_raw_char_length": len(extracted.full_raw),
            }
            doc["pages"] = [
                {
                    "page_no": p.page_no,
                    "extraction_confidence": p.extraction_confidence,
                    "ocr_used": p.ocr_used,
                    "chars": len(p.normalized_text),
                }
                for p in extracted.pages
            ]

            out_name = f"{full_doc_id}.json"
            (OUT_DOCS / out_name).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            all_chunks_flat.extend(doc_chunks)

            registry_rows.append(
                {
                    "doc_id": full_doc_id,
                    "pdf_doc_id": extracted.doc_id,
                    "source_path": rel,
                    "file_name": file_name,
                    "title": doc.get("title") or "",
                    "logical_index": doc.get("logical_index") or "",
                    "chunks": len(doc_chunks),
                    "tables": len(tables_merged),
                    "pages": len(extracted.pages),
                }
            )

        print(f"OK {rel} ({len(logical_parts)} лог. док.)")

    chunks_path = OUT_CHUNKS / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in all_chunks_flat:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    (OUT_TABLES / "tables.json").write_text(
        json.dumps(all_tables_flat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ent = _aggregate_entities(all_chunks_flat, all_tables_flat)
    (OUT_ENTITIES / "entities.json").write_text(
        json.dumps(ent, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reg_csv = OUT_REGISTRY / "index.csv"
    with reg_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "pdf_doc_id",
                "source_path",
                "file_name",
                "title",
                "logical_index",
                "chunks",
                "tables",
                "pages",
            ],
        )
        w.writeheader()
        for row in registry_rows:
            w.writerow(row)

    print(
        f"Готово: записей реестра: {len(registry_rows)}, чанков: {len(all_chunks_flat)}, "
        f"таблиц: {len(all_tables_flat)} → {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
