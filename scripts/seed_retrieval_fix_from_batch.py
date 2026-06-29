#!/usr/bin/env python3
"""Seed retrieval_fix из batch report.json + gold catalog.

Читает ml/experiments/**/report.json, l2_weak_report.json и data/ml/retrieval_fix_gold.json.
Пишет data/ml/feedback/retrieval_fix_seed_batch.jsonl (append-safe, dedupe).

Пример:
  python3 scripts/seed_retrieval_fix_from_batch.py
  python3 scripts/seed_retrieval_fix_from_batch.py --dry-run
  python3 scripts/export_training_feedback.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLIENTS = ROOT / "clients_consult"
GOLD_PATH = ROOT / "data/ml/retrieval_fix_gold.json"
STRUCTURED = ROOT / "structured_index.json"
OUT_PATH = ROOT / "data/ml/feedback/retrieval_fix_seed_batch.jsonl"
EXPERIMENTS = ROOT / "ml/experiments"

PEDIATRIC_MARKERS = ("детс_нас", "дет_нас", "д-нас", "детей", "/дет ", "детское", "pediatr")
ADULT_MARKERS = ("взр_нас", "вз_нас", "в-нас", "взросл", "взрослое")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_structured_paths() -> list[str]:
    if not STRUCTURED.is_file():
        return []
    rows = json.loads(STRUCTURED.read_text(encoding="utf-8"))
    return [str(r.get("path") or "") for r in rows if r.get("path")]


def resolve_path_prefix(prefix: str, catalog: list[str]) -> str:
    p = (prefix or "").strip()
    if not p:
        return ""
    if p in catalog:
        return p
    matches = [c for c in catalog if c.startswith(p)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches, key=len)[0]
    return p


def _path_flags(path: str) -> tuple[bool, bool]:
    low = path.lower()
    ped = any(m in low for m in PEDIATRIC_MARKERS)
    adult = any(m in low for m in ADULT_MARKERS)
    return ped, adult


def load_gold() -> dict[str, dict[str, Any]]:
    if not GOLD_PATH.is_file():
        return {}
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return dict(data.get("cases") or {})


def iter_batch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXPERIMENTS.rglob("*.json")):
        name = path.name
        if name not in ("report.json", "l2_weak_report.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if name == "l2_weak_report.json":
            for r in data.get("results") or []:
                if isinstance(r, dict):
                    r = dict(r)
                    r["_source_report"] = str(path.relative_to(ROOT))
                    rows.append(r)
        else:
            for r in data.get("reports") or []:
                if isinstance(r, dict):
                    r = dict(r)
                    r["_source_report"] = str(path.relative_to(ROOT))
                    rows.append(r)
    return rows


def latest_by_case(batch_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in batch_rows:
        cid = str(r.get("case_id") or "").strip()
        if not cid:
            continue
        out[cid] = r
    return out


def extract_paths(row: dict[str, Any], catalog: list[str]) -> list[str]:
    paths: list[str] = []
    for key in ("protocol_paths", "matched_protocols", "retrieval_top"):
        val = row.get(key)
        if isinstance(val, list):
            for p in val:
                if isinstance(p, str) and p.strip():
                    paths.append(resolve_path_prefix(p.strip(), catalog))
        elif isinstance(val, str) and val.strip():
            paths.append(resolve_path_prefix(val.strip(), catalog))
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_query(case_id: str, *, max_len: int = 600) -> str:
    for ext in (".pdf", ".txt"):
        p = CLIENTS / f"{case_id}{ext}"
        if not p.is_file():
            continue
        if ext == ".txt":
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                from pypdf import PdfReader
            except ImportError:
                return case_id
            reader = PdfReader(io.BytesIO(p.read_bytes()))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        text = re.sub(r"\s+", " ", text).strip()
        for kw in ("Диагноз:", "Диагноз", "Жалобы", "Заключение", "МКБ"):
            i = text.find(kw)
            if i >= 0:
                return text[i : i + max_len].strip()
        return text[:max_len].strip() if text else case_id
    return case_id


def pick_rejected(paths: list[str], gold: dict[str, Any]) -> str:
    if not paths:
        return ""
    reject_subs = [str(s) for s in (gold.get("reject_if_substr") or []) if s]
    for p in paths:
        low = p.lower()
        if any(s.lower() in low for s in reject_subs):
            return p
    return paths[0]


def pick_chosen(
    case_id: str,
    paths: list[str],
    gold: dict[str, Any],
    catalog: list[str],
) -> tuple[str, list[str], str]:
    tags = list(gold.get("tags") or [])
    note = str(gold.get("note") or "")

    chosen = str(gold.get("chosen_path") or "").strip()
    if chosen:
        chosen = resolve_path_prefix(chosen, catalog)
        return chosen, tags, note

    rejected = pick_rejected(paths, gold)
    if not rejected:
        return "", tags, note

    ped_rej, adult_rej = _path_flags(rejected)
    slug = rejected.split("/")[1] if "/" in rejected else ""

    if case_id.startswith("pediatr") or ped_rej is False:
        for p in paths[1:] + catalog:
            if slug and slug not in p:
                continue
            ped, adult = _path_flags(p)
            if ped and not adult:
                if "wrong_population" not in tags:
                    tags.append("wrong_population")
                return p, tags, note or "auto: pediatric path"

    if not case_id.startswith("pediatr"):
        for p in catalog:
            if slug and slug not in p:
                continue
            ped, adult = _path_flags(p)
            if adult and not ped:
                if "wrong_population" not in tags:
                    tags.append("wrong_population")
                return p, tags, note or "auto: adult path"

    return "", tags, note


def event_key(ev: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(ev.get("query") or "")[:200],
            str(ev.get("rejected_path") or ""),
            str(ev.get("chosen_path") or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    fb = ROOT / "data/ml/feedback"
    for fp in [path, fb / "retrieval_fix.jsonl", fb / "kz_routing_retrieval_fix_seed.jsonl"]:
        if not fp.is_file():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") == "retrieval_fix":
                keys.add(event_key(ev))
    return keys


def build_events(*, dry_run: bool = False) -> dict[str, Any]:
    catalog = _load_structured_paths()
    gold_cases = load_gold()
    latest = latest_by_case(iter_batch_rows())

    existing = load_existing_keys(OUT_PATH)
    new_events: list[dict[str, Any]] = []
    skipped: list[str] = []

    for case_id, gold in gold_cases.items():
        row = latest.get(case_id)
        paths = extract_paths(row or {}, catalog)
        rejected = str(gold.get("rejected_path") or "").strip()
        if rejected:
            rejected = resolve_path_prefix(rejected, catalog)
        else:
            rejected = pick_rejected(paths, gold)
        chosen, tags, note = pick_chosen(case_id, paths, gold, catalog)

        if not chosen:
            skipped.append(f"{case_id}: no chosen_path")
            continue
        if not rejected:
            skipped.append(f"{case_id}: no rejected_path in batch")
            continue
        if rejected == chosen:
            skipped.append(f"{case_id}: rejected==chosen")
            continue

        query = load_query(case_id)
        ev = {
            "event_type": "retrieval_fix",
            "ts": _utc_now(),
            "reviewer": "seed-batch-gold",
            "source": "batch_seed",
            "query": query,
            "rejected_path": rejected,
            "chosen_path": chosen,
            "tags": tags or ["wrong_protocol"],
            "note": note,
            "case_id": case_id,
            "batch_report": (row or {}).get("_source_report"),
        }
        k = event_key(ev)
        if k in existing:
            skipped.append(f"{case_id}: duplicate")
            continue
        new_events.append(ev)
        existing.add(k)

    if not dry_run and new_events:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("a", encoding="utf-8") as f:
            for ev in new_events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    return {
        "dry_run": dry_run,
        "gold_cases": len(gold_cases),
        "written": len(new_events),
        "output": str(OUT_PATH),
        "skipped": skipped,
        "samples": new_events[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed retrieval_fix from batch + gold catalog")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = build_events(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
