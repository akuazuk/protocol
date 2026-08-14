"""Парсинг скачанных _s.pdf → labels/{reg_id}.json (GCE-first)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from clinical_knowledge.rceth_sync.crawl import load_manifest
from clinical_knowledge.rceth_sync.label_parse import build_label_record
from clinical_knowledge.rceth_sync.paths import labels_dir, manifest_path, pdf_dir
from clinical_knowledge.rceth_sync.status import write_status, write_sync_summary


def _extract_pdf_text(path: Path) -> tuple[str, list[str], str | None]:
    data = path.read_bytes()
    # OCR часто нужен для сканов Refbank; на Mac/GCE включается по умолчанию.
    from clinical_knowledge.text_extract import extract_pdf_text_bytes

    return extract_pdf_text_bytes(data, max_pages=int(os.environ.get("RCETH_PDF_MAX_PAGES", "40") or "40"))


def parse_downloaded_labels(
    *,
    root: Path | None = None,
    limit: int | None = None,
    reg_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Прочитать PDF из pdfs/instr и записать labels/*.json."""
    rows = load_manifest(manifest_path(root))
    if reg_ids:
        want = {r.strip() for r in reg_ids if r.strip()}
        rows = [r for r in rows if r.get("reg_id") in want]
    else:
        rows = [r for r in rows if r.get("url_s") or r.get("has_s_pdf")]
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    out_dir = labels_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    write_status(
        phase="parse",
        status="running",
        done=0,
        total=total,
        message="parse labels",
        root=root,
    )

    ok_n = 0
    needs_human_n = 0
    missing_pdf = 0
    empty_text = 0
    errors = 0
    for i, row in enumerate(rows, start=1):
        rid = str(row.get("reg_id") or "")
        write_status(
            phase="parse",
            status="running",
            done=i - 1,
            total=total,
            message=f"parse {rid}",
            current_reg_id=rid,
            errors=errors,
            root=root,
        )
        pdf_path = pdf_dir(root) / f"{rid}_s.pdf"
        if not pdf_path.is_file():
            missing_pdf += 1
            continue
        try:
            text, warns, err = _extract_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            label = build_label_record(
                reg_id=rid,
                text="",
                meta={**row, "pdf_s": {"url": row.get("url_s"), "sha256": row.get("pdf_s_sha256"), "bytes": row.get("pdf_s_bytes")}},
            )
            label["parse"] = {
                "ok": False,
                "method": "heading_regex_v1",
                "needs_human": True,
                "error": str(exc)[:200],
                "found_keys": [],
            }
            (out_dir / f"{rid}.json").write_text(
                json.dumps(label, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            needs_human_n += 1
            continue
        if not (text or "").strip():
            empty_text += 1
            label = build_label_record(reg_id=rid, text="", meta=row)
            label["parse"]["ok"] = False
            label["parse"]["needs_human"] = True
            label["parse"]["extract_error"] = err or "empty_text"
            label["parse"]["extract_warnings"] = warns[:8]
            (out_dir / f"{rid}.json").write_text(
                json.dumps(label, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            needs_human_n += 1
            continue
        label = build_label_record(
            reg_id=rid,
            text=text,
            meta={
                **row,
                "pdf_s": {
                    "url": row.get("url_s") or "",
                    "sha256": row.get("pdf_s_sha256") or "",
                    "bytes": row.get("pdf_s_bytes") or pdf_path.stat().st_size,
                },
            },
        )
        label["parse"]["extract_warnings"] = warns[:8]
        if err:
            label["parse"]["extract_error"] = err
        if label["parse"].get("ok"):
            ok_n += 1
        else:
            needs_human_n += 1
        (out_dir / f"{rid}.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "ok": errors == 0,
        "manifest_count": len(load_manifest(manifest_path(root))),
        "parsed": ok_n + needs_human_n,
        "parse_ok": ok_n,
        "needs_human": needs_human_n,
        "missing_pdf": missing_pdf,
        "empty_text": empty_text,
        "failed": errors,
        "with_s_pdf": sum(
            1 for r in load_manifest(manifest_path(root)) if r.get("url_s") or r.get("has_s_pdf")
        ),
        "downloaded": sum(
            1 for r in load_manifest(manifest_path(root)) if (pdf_dir(root) / f"{r.get('reg_id')}_s.pdf").is_file()
        ),
    }
    write_sync_summary(summary, root=root)
    write_status(
        phase="parse",
        status="done",
        done=total,
        total=total,
        message=f"parse_ok={ok_n} needs_human={needs_human_n}",
        errors=errors,
        root=root,
        extra={"summary": summary},
    )
    return summary
