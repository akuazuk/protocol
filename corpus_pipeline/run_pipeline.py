#!/usr/bin/env python3
"""
Полный прогон: PDF → документы, чанки, таблицы, сущности, реестр.

  cd Protocol && python3 -m corpus_pipeline.run_pipeline

Переменные: CORPUS_PDF_ROOT, CORPUS_OUTPUT_ROOT, CORPUS_USE_OCR - см. output/README.md
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


def _parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="PDF → чанки, таблицы, карточки.")
    p.add_argument(
        "--changed-only",
        action="store_true",
        help="Обработать только --only-paths и влить в существующий chunks.jsonl.",
    )
    p.add_argument(
        "--only-paths",
        default="",
        help="Список relative path через запятую или файл со строками.",
    )
    return p.parse_args(argv)


def _resolve_only_paths(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    as_file = Path(text)
    if as_file.is_file():
        return [
            ln.strip().replace("\\", "/")
            for ln in as_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return [p.strip().replace("\\", "/") for p in text.split(",") if p.strip()]


def _source_rel(pdf_path: Path) -> str:
    try:
        return str(pdf_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(pdf_path.relative_to(PDF_ROOT)).replace("\\", "/")
        return f"minzdrav_protocols/{rel}"


def _process_pdf(
    pdf_path: Path,
    *,
    all_chunks_flat: list[dict],
    all_tables_flat: list[dict],
    registry_rows: list[dict],
) -> None:
    rel = _source_rel(pdf_path)
    file_name = pdf_path.name
    try:
        extracted = extract_pdf(pdf_path, rel)
    except Exception as e:
        print(f"SKIP {rel}: {e}", file=sys.stderr)
        return

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


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    only_paths = _resolve_only_paths(args.only_paths)
    changed_only = bool(args.changed_only or only_paths)

    if not PDF_ROOT.is_dir():
        raise SystemExit(
            f"Нет каталога с PDF: {PDF_ROOT}\n"
            "Укажите CORPUS_PDF_ROOT или положите файлы в minzdrav_protocols/"
        )

    _ensure_dirs()
    if not changed_only:
        for p in OUT_DOCS.glob("*.json"):
            p.unlink()

    pdfs = sorted(PDF_ROOT.rglob("*.pdf"))
    if only_paths:
        want = {p.replace("\\", "/") for p in only_paths}
        filtered: list[Path] = []
        for pdf_path in pdfs:
            rel = _source_rel(pdf_path)
            if rel in want or pdf_path.name in {Path(w).name for w in want}:
                filtered.append(pdf_path)
        missing = want - {_source_rel(p) for p in filtered}
        if missing:
            print(f"WARN: не найдены PDF: {sorted(missing)[:12]}", file=sys.stderr)
        pdfs = filtered
        if not pdfs:
            raise SystemExit("changed-only: нет PDF для обработки")

    all_chunks_flat: list[dict] = []
    all_tables_flat: list[dict] = []
    registry_rows: list[dict] = []

    for pdf_path in pdfs:
        _process_pdf(
            pdf_path,
            all_chunks_flat=all_chunks_flat,
            all_tables_flat=all_tables_flat,
            registry_rows=registry_rows,
        )

    chunks_path = OUT_CHUNKS / "chunks.jsonl"
    tables_path = OUT_TABLES / "tables.json"
    replace_paths = {_source_rel(p) for p in pdfs}
    if changed_only and chunks_path.is_file():
        from clinical_knowledge.kp_sync.jsonl_merge import (
            load_jsonl,
            merge_jsonl_by_path,
            merge_tables_json,
            write_jsonl,
        )

        old_chunks = load_jsonl(chunks_path)
        merged_chunks = merge_jsonl_by_path(
            old_chunks, all_chunks_flat, replace_paths=replace_paths
        )
        write_jsonl(chunks_path, merged_chunks)
        all_chunks_flat = merged_chunks
        if tables_path.is_file():
            try:
                old_tables = json.loads(tables_path.read_text(encoding="utf-8"))
                if not isinstance(old_tables, list):
                    old_tables = []
            except json.JSONDecodeError:
                old_tables = []
            all_tables_flat = merge_tables_json(
                old_tables, all_tables_flat, replace_paths=replace_paths
            )
    else:
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

    reg_fields = [
        "doc_id",
        "pdf_doc_id",
        "source_path",
        "file_name",
        "title",
        "logical_index",
        "chunks",
        "tables",
        "pages",
    ]
    reg_csv = OUT_REGISTRY / "index.csv"
    if changed_only and reg_csv.is_file():
        with reg_csv.open(encoding="utf-8", newline="") as f:
            old_reg = list(csv.DictReader(f))
        from clinical_knowledge.kp_sync.jsonl_merge import merge_jsonl_by_path

        registry_rows = merge_jsonl_by_path(
            old_reg, registry_rows, replace_paths=replace_paths
        )
    with reg_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reg_fields)
        w.writeheader()
        for row in registry_rows:
            w.writerow({k: row.get(k, "") for k in reg_fields})

    print(
        f"Готово: записей реестра: {len(registry_rows)}, чанков: {len(all_chunks_flat)}, "
        f"таблиц: {len(all_tables_flat)} → {OUTPUT_ROOT}"
        + (" (changed-only merge)" if changed_only else "")
    )

    try:
        from corpus_pipeline.protocol_cards import build_all_protocol_cards, write_protocol_cards_jsonl

        cards = build_all_protocol_cards(
            ROOT,
            documents_dir=OUT_DOCS,
            manifest_path=PDF_ROOT / "_manifest.jsonl",
        )
        cards_path = OUT_REGISTRY / "protocol_cards.jsonl"
        if changed_only and cards_path.is_file():
            from clinical_knowledge.kp_sync.jsonl_merge import load_jsonl, merge_jsonl_by_path

            cards = merge_jsonl_by_path(
                load_jsonl(cards_path), cards, replace_paths=replace_paths
            )
        write_protocol_cards_jsonl(cards, cards_path)
        if not changed_only:
            gastro = [c for c in cards if c.get("specialty_slug") == "gastroenterologiya"]
            gastro_dir = ROOT / "data" / "gastro_mvp"
            gastro_dir.mkdir(parents=True, exist_ok=True)
            write_protocol_cards_jsonl(gastro, gastro_dir / "protocol_registry.jsonl")
        print(f"Карточки протоколов: {len(cards)} → {cards_path}")
    except Exception as e:
        print(f"WARN: protocol_cards не собраны: {e}")


if __name__ == "__main__":
    main()
