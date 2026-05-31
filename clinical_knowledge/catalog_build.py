"""Сборка path/corpus-правил по всему каталогу protocol_cards (все рубрики)."""
from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalog"
CATALOG_RULES_DIR = CATALOG_DIR / "rules"
COVERAGE_PATH = CATALOG_DIR / "rules_coverage_report.json"
DEFAULT_REGISTRY = ROOT / "output" / "registry" / "protocol_cards.jsonl"
DEFAULT_CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"


def catalog_registry_path() -> Path:
    reg = DEFAULT_REGISTRY
    if reg.is_file():
        return reg
    fallback = ROOT / "data" / "gastro_mvp" / "protocol_registry.jsonl"
    return fallback if fallback.is_file() else reg


def catalog_source_paths(
    registry_jsonl: Path | None = None,
    *,
    specialty_slug: str | None = None,
) -> list[str]:
    """Уникальные source_path из protocol_cards (опционально одна рубрика)."""
    path = registry_jsonl or catalog_registry_path()
    if not path.is_file():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if specialty_slug and row.get("specialty_slug") != specialty_slug:
            continue
        sp = (row.get("source_path") or "").replace("\\", "/")
        if sp and sp not in seen:
            seen.add(sp)
            out.append(sp)
    return sorted(out)


def build_chunks_index(chunks_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Один проход по JSONL: source_path → чанки."""
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not chunks_path.is_file():
        return index
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = (c.get("source_path") or "").replace("\\", "/")
            if sp:
                index[sp].append(c)
    return index


def extract_rules_all_catalog_pdfs(
    chunks_path: Path,
    registry_jsonl: Path | None = None,
    *,
    chunks_index: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Извлечь правила по всем уникальным PDF каталога (все рубрики)."""
    from .rules_from_corpus import extract_rules_from_chunks, pick_best_logical_chunks
    from .rules_from_path import extract_path_rules

    pdfs = catalog_source_paths(registry_jsonl)
    if chunks_index is None:
        chunks_index = build_chunks_index(chunks_path)

    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_pdf: dict[str, Any] = {}
    by_rubric: dict[str, dict[str, int]] = defaultdict(lambda: {"pdfs": 0, "with_rules": 0})

    for sp in pdfs:
        rubric = sp.split("/")[1] if sp.startswith("minzdrav_protocols/") and "/" in sp[18:] else ""
        if not rubric and "/" in sp:
            parts = sp.replace("\\", "/").split("/")
            rubric = parts[1] if len(parts) > 1 else "unknown"
        by_rubric[rubric]["pdfs"] += 1

        all_ch = chunks_index.get(sp) or []
        if not all_ch:
            per_pdf[sp] = {"chunks": 0, "rules": 0, "skipped": "no_chunks", "rubric": rubric}
            continue

        doc_chunks = pick_best_logical_chunks(all_ch)
        pdf_hash = sha256(sp.encode()).hexdigest()[:8]
        protocol_id = f"proto_{pdf_hash}"
        extracted = extract_rules_from_chunks(
            doc_chunks,
            protocol_id=protocol_id,
            rule_id_prefix=pdf_hash,
            source_path=sp,
        )
        n_rules = sum(len(v) for v in extracted.values())
        extraction_method = "corpus_heuristic" if n_rules else None

        if n_rules == 0:
            path_rules = extract_path_rules(
                sp, protocol_id=protocol_id, rule_id_prefix=pdf_hash
            )
            for cid, rules in path_rules.items():
                extracted.setdefault(cid, []).extend(rules)
            n_rules = sum(len(v) for v in extracted.values())
            if n_rules:
                extraction_method = "path_template"

        if n_rules:
            by_rubric[rubric]["with_rules"] += 1

        per_pdf[sp] = {
            "chunks": len(doc_chunks),
            "rules": n_rules,
            "rubric": rubric,
            "doc_id": doc_chunks[0].get("doc_id") if doc_chunks else None,
            "extraction_method": extraction_method,
        }
        for cid, rules in extracted.items():
            merged[cid].extend(rules)

    rubric_summary = {
        slug: {
            "pdfs_total": v["pdfs"],
            "pdfs_with_rules": v["with_rules"],
            "coverage_pct": round(100.0 * v["with_rules"] / v["pdfs"], 1) if v["pdfs"] else 0.0,
        }
        for slug, v in sorted(by_rubric.items())
    }

    meta = {
        "pdfs_total": len(pdfs),
        "pdfs_with_rules": sum(1 for p in per_pdf.values() if isinstance(p, dict) and (p.get("rules") or 0) > 0),
        "pdfs": per_pdf,
        "by_rubric": rubric_summary,
        "scope": "all_catalog",
    }
    return dict(merged), meta


def merge_rules_into_catalog(
    extracted: dict[str, list[dict[str, Any]]],
    out_dir: Path | None = None,
) -> dict[str, int]:
    """Записать data/catalog/rules/auto_<cid>.json и path_<cid>.json."""
    out_dir = out_dir or CATALOG_RULES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    auto_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cid, rules in extracted.items():
        for r in rules:
            if r.get("extraction_method") == "path_template":
                path_by_cid[cid].append(r)
            else:
                auto_by_cid[cid].append(r)

    def _write_bucket(prefix: str, bucket: dict[str, list[dict[str, Any]]]) -> None:
        for cid, rules in bucket.items():
            fpath = out_dir / f"{prefix}_{cid}.json"
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for r in rules:
                rid = str(r.get("rule_id") or "")
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                deduped.append(r)
            if not deduped:
                continue
            payload = {
                "condition_id": cid,
                "rules": deduped,
                f"{prefix}_only": True,
                "catalog_scope": "all_rubrics",
            }
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            counts[cid] = counts.get(cid, 0) + len(deduped)

    _write_bucket("auto", auto_by_cid)
    _write_bucket("path", path_by_cid)
    return counts


def write_coverage_report(meta: dict[str, Any], extracted: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Сохранить rules_coverage_report.json с разбивкой по рубрикам."""
    with_rules = [
        sp for sp, info in (meta.get("pdfs") or {}).items()
        if isinstance(info, dict) and (info.get("rules") or 0) > 0
    ]
    without_rules = [
        sp for sp, info in (meta.get("pdfs") or {}).items()
        if isinstance(info, dict) and (info.get("rules") or 0) == 0
    ]
    report = {
        "pdfs_total": meta.get("pdfs_total"),
        "pdfs_with_rules": len(with_rules),
        "pdfs_without_rules": len(without_rules),
        "total_rules": sum(len(v) for v in extracted.values()),
        "rules_by_condition": {cid: len(rules) for cid, rules in extracted.items()},
        "by_rubric": meta.get("by_rubric") or {},
        "with_rules": with_rules,
        "without_rules": without_rules,
        "per_pdf": meta.get("pdfs"),
        "scope": "all_catalog",
    }
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_catalog_rules(
    *,
    chunks_path: Path | None = None,
    registry_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Полный цикл: extract → merge → coverage report."""
    cp = chunks_path or DEFAULT_CHUNKS
    reg = registry_jsonl or catalog_registry_path()
    index = build_chunks_index(cp)
    extracted, meta = extract_rules_all_catalog_pdfs(cp, reg, chunks_index=index)
    rule_counts = merge_rules_into_catalog(extracted)
    report = write_coverage_report(meta, extracted)
    return {
        "pdfs_total": report.get("pdfs_total"),
        "pdfs_with_rules": report.get("pdfs_with_rules"),
        "coverage_pct": round(
            100.0 * int(report.get("pdfs_with_rules") or 0) / max(1, int(report.get("pdfs_total") or 1)),
            1,
        ),
        "conditions": len(rule_counts),
        "total_rules": report.get("total_rules"),
        "rubrics": len(report.get("by_rubric") or {}),
        "report_path": str(COVERAGE_PATH.relative_to(ROOT)),
    }
