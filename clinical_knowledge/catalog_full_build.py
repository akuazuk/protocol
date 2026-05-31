"""Полная структуризация каталога (478 PDF): conditions + rules как gastro MVP, с прогрессом %."""
from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalog"
CONDITIONS_DIR = CATALOG_DIR / "conditions"
ENRICH_DIR = CATALOG_DIR / "enrichment"
BUILD_STATE_PATH = CATALOG_DIR / "build_state.json"

ProgressFn = Callable[[dict[str, Any]], None]


def _cards_by_source_path() -> dict[str, dict[str, Any]]:
    from .catalog_build import catalog_registry_path

    reg = catalog_registry_path()
    out: dict[str, dict[str, Any]] = {}
    if not reg.is_file():
        return out
    for line in reg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = (row.get("source_path") or "").replace("\\", "/")
        if sp and sp not in out:
            out[sp] = row
    return out


def _rules_for_pdf(extracted: dict[str, list[dict[str, Any]]], source_path: str) -> list[dict[str, Any]]:
    sp = source_path.replace("\\", "/")
    out: list[dict[str, Any]] = []
    for rules in extracted.values():
        for r in rules:
            src = (r.get("source") or {}).get("source_path") or ""
            if src.replace("\\", "/") == sp:
                out.append(r)
    return out


def _condition_ids_for_pdf(extracted: dict[str, list[dict[str, Any]]], source_path: str) -> list[str]:
    sp = source_path.replace("\\", "/")
    cids: list[str] = []
    for cid, rules in extracted.items():
        if any((r.get("source") or {}).get("source_path", "").replace("\\", "/") == sp for r in rules):
            cids.append(cid)
    return cids


def _pick_primary_condition_id(
    source_path: str,
    extracted: dict[str, list[dict[str, Any]]],
    card: dict[str, Any] | None,
) -> str:
    if extracted:
        return max(extracted.keys(), key=lambda k: len(extracted.get(k) or []))
    from .rules_from_path import infer_path_condition

    hit = infer_path_condition(source_path)
    if hit:
        return hit[0]
    parts = source_path.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0] == "minzdrav_protocols":
        slug = parts[1]
        return f"rubric_{slug.replace('-', '_')}"
    if card and card.get("title"):
        from .condition_builder import slug_condition_from_title

        return slug_condition_from_title(str(card.get("title")))
    return f"pdf_{sha256(source_path.encode()).hexdigest()[:10]}"


def _emit(
    on_progress: ProgressFn | None,
    *,
    stage: str,
    pct: int,
    label_ru: str,
    partial: dict[str, Any] | None = None,
) -> None:
    if not on_progress:
        return
    on_progress(
        {
            "type": "progress",
            "stage": stage,
            "pct": pct,
            "label_ru": label_ru,
            "partial": partial or {},
        }
    )


def _write_build_state(state: dict[str, Any]) -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    BUILD_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_build_state() -> dict[str, Any]:
    if not BUILD_STATE_PATH.is_file():
        return {}
    try:
        data = json.loads(BUILD_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_status_payload() -> dict[str, Any]:
    """Статус для API / UI: % готовности каталога и промежуточные метрики."""
    from .coverage import coverage_status_payload

    state = load_build_state()
    cov = coverage_status_payload()
    pdfs_total = int(state.get("pdfs_total") or cov.get("pdfs_total") or 0)
    pdfs_structured = int(state.get("pdfs_structured") or cov.get("pdfs_with_rules") or 0)
    conditions_n = int(state.get("conditions_written") or 0)
    build_pct = round(100.0 * pdfs_structured / pdfs_total, 1) if pdfs_total else 0.0
    return {
        "build_complete": bool(state.get("complete")),
        "build_pct": build_pct,
        "pdfs_total": pdfs_total,
        "pdfs_structured": pdfs_structured,
        "pdfs_without_structure": max(0, pdfs_total - pdfs_structured),
        "conditions_written": conditions_n,
        "rules_total": int(state.get("rules_total") or cov.get("total_auto_rules") or 0),
        "rules_coverage_pct": cov.get("coverage_pct"),
        "by_rubric": cov.get("by_rubric") or {},
        "last_stage": state.get("last_stage"),
        "updated_utc": state.get("updated_utc") or state.get("finished_utc"),
        "llm_enriched": int(state.get("llm_enriched") or 0),
    }


def write_conditions_catalog(
    conditions: dict[str, dict[str, Any]],
    out_dir: Path | None = None,
) -> int:
    out_dir = out_dir or CONDITIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cid, record in sorted(conditions.items()):
        path = out_dir / f"{cid}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
    return written


def build_catalog_full(
    *,
    chunks_path: Path | None = None,
    registry_jsonl: Path | None = None,
    on_progress: ProgressFn | None = None,
    use_llm: bool = False,
    llm_limit: int = 0,
) -> dict[str, Any]:
    """Полный цикл: rules + gastro-like conditions по всем PDF каталога."""
    from .catalog_build import (
        build_chunks_index,
        catalog_registry_path,
        catalog_source_paths,
        extract_rules_all_catalog_pdfs,
        merge_rules_into_catalog,
        write_coverage_report,
    )
    from .condition_builder import build_condition_record, merge_condition_records

    cp = chunks_path or (ROOT / "output" / "chunks" / "chunks.jsonl")
    reg = registry_jsonl or catalog_registry_path()
    pdfs = catalog_source_paths(reg)
    cards_map = _cards_by_source_path()

    state: dict[str, Any] = {
        "complete": False,
        "pdfs_total": len(pdfs),
        "pdfs_structured": 0,
        "conditions_written": 0,
        "rules_total": 0,
        "llm_enriched": 0,
        "last_stage": "init",
    }
    _write_build_state(state)

    _emit(
        on_progress,
        stage="index",
        pct=2,
        label_ru="Индекс чанков…",
        partial={"pdfs_total": len(pdfs)},
    )
    index = build_chunks_index(cp)

    _emit(on_progress, stage="extract", pct=5, label_ru="Извлечение правил по PDF…")
    extracted, meta = extract_rules_all_catalog_pdfs(cp, reg, chunks_index=index)
    merge_rules_into_catalog(extracted)
    report = write_coverage_report(meta, extracted)

    conditions: dict[str, dict[str, Any]] = {}
    per_pdf: dict[str, Any] = {}
    llm_model = None
    llm_generate = None
    llm_extract = None
    llm_done = 0

    if use_llm:
        try:
            from rag_server import _extract_gemini_text, generate_gemini, get_gemini

            llm_model = get_gemini()
            llm_generate = generate_gemini
            llm_extract = _extract_gemini_text
        except Exception:
            use_llm = False

    total = max(1, len(pdfs))
    for i, sp in enumerate(pdfs):
        pct = 10 + int(75 * (i + 1) / total)
        info = (meta.get("pdfs") or {}).get(sp) or {}
        n_rules = int(info.get("rules") or 0)
        rules_for_pdf = _rules_for_pdf(extracted, sp)
        cids = _condition_ids_for_pdf(extracted, sp)
        card = cards_map.get(sp) or {
            "source_path": sp,
            "title": Path(sp).stem,
            "specialty_slug": sp.split("/")[1] if "/" in sp else "",
            "protocol_id": f"proto_{sha256(sp.encode()).hexdigest()[:8]}",
        }
        cid = cids[0] if cids else _pick_primary_condition_id(sp, {}, card)

        enrichment_payload = None
        if use_llm and n_rules == 0 and llm_model and (llm_limit <= 0 or llm_done < llm_limit):
            from .enrichment_samples import sample_text_for_pdf
            from .llm_enrich import enrich_condition_text

            sample = sample_text_for_pdf(sp, cp)
            if sample:
                enrichment_payload = enrich_condition_text(
                    cid,
                    sample,
                    model=llm_model,
                    generate_fn=llm_generate,
                    extract_text_fn=llm_extract,
                )
                if enrichment_payload:
                    enrichment_payload["source_path"] = sp
                    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
                    cache_name = f"{cid}_{enrichment_payload.get('text_hash', 'x')}.json"
                    (ENRICH_DIR / cache_name).write_text(
                        json.dumps(enrichment_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    from .rules_from_enrichment import enrichment_payload_to_rules

                    for rule in enrichment_payload_to_rules(enrichment_payload):
                        extracted.setdefault(cid, []).append(rule)
                        rules_for_pdf.append(rule)
                    llm_done += 1
                    n_rules = len(rules_for_pdf)

        record = build_condition_record(cid, card, rules_for_pdf, enrichment=enrichment_payload)
        if cid in conditions:
            conditions[cid] = merge_condition_records(conditions[cid], record)
        else:
            conditions[cid] = record

        n_rules = len(rules_for_pdf)
        structured = n_rules > 0 or bool(enrichment_payload)
        per_pdf[sp] = {
            "condition_id": cid,
            "rules": n_rules,
            "structured": structured,
            "rubric": info.get("rubric"),
            "extraction_method": info.get("extraction_method"),
        }

        pdfs_structured = sum(1 for p in per_pdf.values() if p.get("structured"))
        state.update(
            {
                "pdfs_structured": pdfs_structured,
                "conditions_written": len(conditions),
                "rules_total": sum(len(v) for v in extracted.values()),
                "llm_enriched": llm_done,
                "last_stage": "pdf",
                "build_pct": round(100.0 * pdfs_structured / total, 1),
            }
        )
        if i % 5 == 0 or i + 1 == total:
            _write_build_state(state)
            _emit(
                on_progress,
                stage="pdf",
                pct=pct,
                label_ru=f"Структуризация PDF {i + 1}/{total}",
                partial={
                    "pdfs_done": i + 1,
                    "pdfs_total": total,
                    "pdfs_structured": pdfs_structured,
                    "build_pct": state["build_pct"],
                    "conditions": len(conditions),
                    "current_pdf": Path(sp).name[:80],
                    "rules_coverage_pct": report.get("pdfs_with_rules")
                    and round(100.0 * int(report["pdfs_with_rules"]) / total, 1),
                },
            )

    _emit(on_progress, stage="conditions", pct=88, label_ru="Запись conditions JSON…")
    cond_n = write_conditions_catalog(conditions)

    if llm_done:
        merge_rules_into_catalog(extracted)

    _emit(on_progress, stage="report", pct=95, label_ru="Отчёт покрытия…")
    meta["pdfs_with_rules"] = sum(1 for p in per_pdf.values() if p.get("structured"))
    report = write_coverage_report(meta, extracted)

    summary = {
        "pdfs_total": len(pdfs),
        "pdfs_structured": sum(1 for p in per_pdf.values() if p.get("structured")),
        "coverage_pct": round(
            100.0 * sum(1 for p in per_pdf.values() if p.get("structured")) / max(1, len(pdfs)),
            1,
        ),
        "conditions_written": cond_n,
        "rules_total": report.get("total_rules"),
        "rubrics": len(report.get("by_rubric") or {}),
        "llm_enriched": llm_done,
        "conditions_dir": str(CONDITIONS_DIR.relative_to(ROOT)),
        "report_path": str((CATALOG_DIR / "rules_coverage_report.json").relative_to(ROOT)),
    }

    state.update(
        {
            "complete": True,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_stage": "done",
            "build_pct": summary["coverage_pct"],
            "pdfs_structured": summary["pdfs_structured"],
            "conditions_written": cond_n,
            "rules_total": summary["rules_total"],
            "llm_enriched": llm_done,
            "summary": summary,
        }
    )
    _write_build_state(state)

    _emit(
        on_progress,
        stage="done",
        pct=100,
        label_ru="Структуризация каталога завершена",
        partial=summary,
    )
    return summary
