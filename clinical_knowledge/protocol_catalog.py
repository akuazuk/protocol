"""Единый каталог PDF протоколов: МКБ, аудитория, рубрика (478 КП)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "protocol_catalog.jsonl"
OVERRIDES_PATH = ROOT / "data" / "protocol_audience_overrides.json"
INDEX_CSV = ROOT / "index.csv"
CARDS_PATH = ROOT / "output" / "registry" / "protocol_cards.jsonl"
CHUNKS_PATH = ROOT / "output" / "chunks" / "chunks.jsonl"
SUMMARY_JSON_DIR = ROOT / "data" / "protocol_summaries" / "json"
ICD_REF_PATH = ROOT / "data" / "icd_reference" / "icd10_who_2016_terminal_codes.json"
PRIMARY_ICD_LIMIT = 12

_PED_SLUGS = frozenset({"pediatriya"})
_PED_MARKERS = (
    "д-нас", "дет-нас", "дет нас", "дет_нас", "детс", "дет. нас", "детск",
    "детей", " дет", " неонат", "новорожд", "pediatr", "дет возраста",
)
_ADULT_MARKERS = (
    "взросл", "взр ", "взр.", "взр_нас", "в-нас", " в нас", "вз н", "вз_н",
)


def _infer_audience_from_text(path: str, title: str) -> str | None:
    blob = re.sub(r"\s+", " ", f"{path} {title}".lower().replace("_", " ").replace("-", " ")).strip()
    has_p = any(m in blob for m in _PED_MARKERS)
    has_a = any(m in blob for m in _ADULT_MARKERS)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None
CATALOG_PATH = ROOT / "data" / "protocol_catalog.jsonl"
OVERRIDES_PATH = ROOT / "data" / "protocol_audience_overrides.json"
INDEX_CSV = ROOT / "index.csv"
CARDS_PATH = ROOT / "output" / "registry" / "protocol_cards.jsonl"
CHUNKS_PATH = ROOT / "output" / "chunks" / "chunks.jsonl"
SUMMARY_JSON_DIR = ROOT / "data" / "protocol_summaries" / "json"
ICD_REF_PATH = ROOT / "data" / "icd_reference" / "icd10_who_2016_terminal_codes.json"
PRIMARY_ICD_LIMIT = 12

_PED_SLUGS = frozenset({"pediatriya"})


def _norm_path(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@lru_cache(maxsize=1)
def _valid_icd_codes() -> frozenset[str]:
    if not ICD_REF_PATH.is_file():
        return frozenset()
    try:
        data = json.loads(ICD_REF_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    codes: set[str] = set()
    for row in data if isinstance(data, list) else []:
        c = _norm_icd(str(row.get("code") or ""))
        if c:
            codes.add(c)
    return frozenset(codes)


def _filter_valid_icd(codes: list[str]) -> list[str]:
    valid = _valid_icd_codes()
    roots = {c[:3] for c in valid if len(c) >= 3}
    out: list[str] = []
    seen: set[str] = set()
    for raw in codes:
        c = _norm_icd(raw)
        if not c or len(c) < 3 or c in seen:
            continue
        if valid:
            if c in valid:
                seen.add(c)
                out.append(c)
                continue
            root = c[:3]
            if root in roots:
                seen.add(c)
                out.append(c)
                continue
            continue
        seen.add(c)
        out.append(c)
    return out


def _load_overrides() -> dict[str, str]:
    if not OVERRIDES_PATH.is_file():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {_norm_path(k): str(v) for k, v in data.items() if v}


def _cards_by_path() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for card in _read_jsonl(CARDS_PATH):
        sp = _norm_path(str(card.get("source_path") or ""))
        if not sp:
            continue
        if sp not in merged:
            merged[sp] = dict(card)
            continue
        icd = set(merged[sp].get("icd10_all") or [])
        icd |= {str(x) for x in (card.get("icd10_all") or []) if x}
        merged[sp]["icd10_all"] = sorted(icd)
        if not merged[sp].get("icd10_primary") and card.get("icd10_primary"):
            merged[sp]["icd10_primary"] = card.get("icd10_primary")
    return merged


def _icd_from_chunks() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    freq: dict[str, Counter[str]] = defaultdict(Counter)
    if not CHUNKS_PATH.is_file():
        return {}, {}
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = _norm_path(str(c.get("source_path") or ""))
        if not sp:
            continue
        for code in c.get("icd10_codes") or []:
            cc = _norm_icd(str(code))
            if len(cc) >= 3:
                freq[sp][cc] += 1
    all_codes = {sp: sorted(cnt) for sp, cnt in freq.items()}
    by_freq = {sp: [code for code, _ in cnt.most_common()] for sp, cnt in freq.items()}
    return all_codes, by_freq


def _icd_from_summaries() -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if not SUMMARY_JSON_DIR.is_dir():
        return {}
    for path in sorted(SUMMARY_JSON_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = data.get("source") if isinstance(data.get("source"), dict) else {}
        lp = _norm_path(str((src or {}).get("local_path") or ""))
        if not lp:
            continue
        for cond in data.get("conditions") or []:
            if not isinstance(cond, dict):
                continue
            for raw in cond.get("icd10_codes") or []:
                c = _norm_icd(str(raw))
                if c:
                    out[lp].add(c)
    return {k: sorted(v) for k, v in out.items()}


def _infer_audience(
    path: str,
    title: str,
    specialty_slug: str,
    card_population: str | None,
    override: str | None,
) -> tuple[str, str]:
    if override in ("adult", "pediatric", "mixed", "any"):
        return override, "override"
    aud = _infer_audience_from_text(path, title)
    if aud == "pediatric":
        return "pediatric", "filename"
    if aud == "adult":
        return "adult", "filename"
    if aud == "mixed":
        return "mixed", "filename"
    pop = (card_population or "").strip().lower()
    if pop == "child":
        return "pediatric", "card_population"
    if pop == "adult":
        return "adult", "card_population"
    if specialty_slug in _PED_SLUGS and "взросл" not in title.lower():
        return "pediatric", "specialty_slug"
    return "any", "default"


def build_protocol_catalog(*, write: bool = True) -> list[dict[str, Any]]:
    """Собрать каталог из index.csv, cards, chunks, summaries."""
    import importlib.util

    ent_path = ROOT / "corpus_pipeline" / "entities_extract.py"
    spec = importlib.util.spec_from_file_location("entities_extract", ent_path)
    assert spec and spec.loader
    ent_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ent_mod)
    extract_icd10 = ent_mod.extract_icd10

    diag_path = ROOT / "clinical_knowledge" / "diagnosis_icd.py"
    dspec = importlib.util.spec_from_file_location("diagnosis_icd", diag_path)
    assert dspec and dspec.loader
    diag_mod = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(diag_mod)
    lookup_disease_icd = diag_mod.lookup_disease_icd

    if not INDEX_CSV.is_file():
        raise FileNotFoundError(f"Missing {INDEX_CSV}")

    cards = _cards_by_path()
    chunk_all, chunk_freq = _icd_from_chunks()
    summary_icd = _icd_from_summaries()
    overrides = _load_overrides()

    rows: list[dict[str, Any]] = []
    with INDEX_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = _norm_path(row.get("relative_path") or "")
            if not path:
                continue
            filename = row.get("filename") or Path(path).name
            title = row.get("display_title") or re.sub(r"\.pdf$", "", filename, flags=re.I)
            specialty = row.get("category") or ""
            card = cards.get(path) or {}

            sources: set[str] = set()
            icd_set: set[str] = set()
            for src, codes in (
                ("card", card.get("icd10_all") or card.get("icd10_primary") or []),
                ("chunks", chunk_all.get(path, [])),
                ("summary", summary_icd.get(path, [])),
                ("filename", extract_icd10(f"{filename} {title}")),
                ("nomenclature", lookup_disease_icd(f"{filename} {title}")),
            ):
                for raw in codes:
                    c = _norm_icd(str(raw))
                    if c:
                        icd_set.add(c)
                        sources.add(src)

            icd_all = _filter_valid_icd(sorted(icd_set))
            primary_src = (
                [ _norm_icd(x) for x in (card.get("icd10_primary") or []) if x ]
                or chunk_freq.get(path, [])[:PRIMARY_ICD_LIMIT]
                or icd_all[:PRIMARY_ICD_LIMIT]
            )
            icd_primary = _filter_valid_icd(list(dict.fromkeys(primary_src)))[:PRIMARY_ICD_LIMIT]

            aud_csv = (row.get("audience") or "").strip()
            override = overrides.get(path)
            if aud_csv in ("adult", "pediatric", "mixed"):
                audience, aud_src = aud_csv, "index_csv"
            else:
                audience, aud_src = _infer_audience(
                    path,
                    title,
                    specialty,
                    str(card.get("population") or ""),
                    override,
                )

            confidence = "high" if icd_primary else ("medium" if icd_all else "low")
            entry = {
                "path": path,
                "specialty_slug": specialty,
                "display_title": title,
                "audience": audience,
                "audience_source": aud_src,
                "icd10_primary": icd_primary,
                "icd10_all": icd_all,
                "icd_sources": sorted(sources),
                "icd_count": len(icd_all),
                "confidence": confidence,
                "general_scope": not icd_all,
            }
            rows.append(entry)

    if write:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CATALOG_PATH.open("w", encoding="utf-8") as out:
            for entry in rows:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return rows


@lru_cache(maxsize=1)
def load_protocol_catalog() -> dict[str, dict[str, Any]]:
    if not CATALOG_PATH.is_file():
        try:
            build_protocol_catalog(write=True)
        except Exception:
            return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(CATALOG_PATH):
        sp = _norm_path(str(row.get("path") or ""))
        if sp:
            out[sp] = row
    return out


def catalog_stats() -> dict[str, Any]:
    cat = load_protocol_catalog()
    total = len(cat)
    with_icd = sum(1 for r in cat.values() if r.get("icd10_all"))
    with_aud = sum(1 for r in cat.values() if r.get("audience") not in ("", "any", None))
    return {
        "total": total,
        "with_icd": with_icd,
        "without_icd": total - with_icd,
        "with_explicit_audience": with_aud,
    }
