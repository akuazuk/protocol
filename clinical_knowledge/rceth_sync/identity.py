"""Identity Rceth: trade_name → INN / форма / ATC (без override курируемого словаря)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Iterable

from clinical_knowledge.rceth_sync.crawl import load_manifest
from clinical_knowledge.rceth_sync.paths import ROOT, data_root, manifest_path

SEED_PATH = ROOT / "data" / "drug_safety" / "rceth_identity_seed.json"

_COMBO_RE = re.compile(r"[+/]|\s+и\s+|\s+&\s+|,\s+")
_CYR_RE = re.compile(r"[а-яё]", re.I)
_LATIN_INN_RE = re.compile(r"[a-z][a-z0-9 \-]{1,60}$")

# Типичные дыры транслитерации RU INN → канон DDInter.
_INN_ALIASES = {
    "nimesulid": "nimesulide",
    "ketorolak": "ketorolac",
    "diclofenac": "diclofenac",
    "ibuprofen": "ibuprofen",
    "levothyroxin": "levothyroxine",
    "levothyroxine": "levothyroxine",
}

_FORM_MAP: tuple[tuple[str, str], ...] = (
    ("лиофил", "lyophilisate"),
    ("суспенз", "suspension"),
    ("суппозит", "suppository"),
    ("таблет", "tablet"),
    ("капсул", "capsule"),
    ("раствор", "solution"),
    ("порош", "powder"),
    ("эмульс", "emulsion"),
    ("спрей", "spray"),
    ("сироп", "syrup"),
    ("мазь", "ointment"),
    ("крем", "cream"),
    ("гель", "gel"),
    ("капл", "drops"),
)


def _fold(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("ё", "е").split())


def is_combo_inn(raw: str) -> bool:
    s = _fold(raw)
    if not s:
        return False
    if _COMBO_RE.search(s):
        return True
    return len(s.split()) > 3 or len(s) > 80


def form_keywords(form_text: str) -> list[str]:
    blob = _fold(form_text)
    found: list[str] = []
    for needle, key in _FORM_MAP:
        if needle in blob and key not in found:
            found.append(key)
    return found


def canon_inn(raw: str) -> str | None:
    """Сырой INN с карточки Refbank → один латинский канон или None (combo/мусор)."""
    s = _fold(raw)
    if not s or is_combo_inn(s):
        return None
    if _CYR_RE.search(s):
        from clinical_knowledge.drug_normalizer import transliterate

        s = transliterate(s)
    s = _fold(s)
    s = _INN_ALIASES.get(s, s)
    if not _LATIN_INN_RE.match(s):
        return None
    return s


def build_identity_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """trade_norm → {inn, forms, atc, source}. Коллизия двух INN на бренд - ключ выкидываем."""
    acc: dict[str, dict[str, Any]] = {}
    banned: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "active")
        if status and status != "active":
            continue
        trade = _fold(str(row.get("trade_name_ru") or row.get("trade_name") or ""))
        inn = canon_inn(str(row.get("inn") or ""))
        if not trade or len(trade) < 3 or not inn:
            continue
        if trade in banned:
            continue
        forms = form_keywords(str(row.get("form_text") or ""))
        extra_forms = row.get("forms")
        if isinstance(extra_forms, list):
            for item in extra_forms:
                key = _fold(str(item))
                if key and key not in forms:
                    forms.append(key)
        atc = str(row.get("atc") or "").strip()
        prev = acc.get(trade)
        if prev and prev.get("inn") != inn:
            banned.add(trade)
            acc.pop(trade, None)
            continue
        if prev:
            merged_forms = list(prev.get("forms") or [])
            for item in forms:
                if item not in merged_forms:
                    merged_forms.append(item)
            prev["forms"] = merged_forms
            if atc and not prev.get("atc"):
                prev["atc"] = atc
            continue
        acc[trade] = {
            "inn": inn,
            "forms": forms,
            "atc": atc,
            "source": "rceth",
        }
    return acc


def merge_brand_overrides(curated: dict[str, str], rceth: dict[str, str]) -> dict[str, str]:
    """Курируемый словарь побеждает; Rceth только setdefault."""
    out = {_fold(k): _fold(v) for k, v in curated.items() if k and v}
    for key, inn in rceth.items():
        k, v = _fold(key), _fold(inn)
        if k and v:
            out.setdefault(k, v)
    return out


def _seed_rows() -> list[dict[str, Any]]:
    if not SEED_PATH.is_file():
        return []
    try:
        data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


@lru_cache(maxsize=8)
def load_identity_index(root: str | None = None) -> dict[str, dict[str, Any]]:
    """Seed из git + манифест с data volume (GCE), если есть."""
    rows = list(_seed_rows())
    data = data_root(root) if root else data_root()
    man = manifest_path(data)
    if man.is_file():
        rows.extend(load_manifest(man))
    return build_identity_index(rows)


def rceth_brand_to_inn(root: str | None = None) -> dict[str, str]:
    return {
        brand: str(rec["inn"])
        for brand, rec in load_identity_index(root).items()
        if rec.get("inn")
    }


def clear_identity_cache() -> None:
    load_identity_index.cache_clear()
