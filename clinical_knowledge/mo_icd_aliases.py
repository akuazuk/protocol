"""Клинические алиасы диагноза для сверки со справочником МКБ.

Алиасы расширяют query для name_match / directory suggest.
seed_codes - мягкая подсказка кандидатов, сами по себе chip ok не ставят.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_ALIASES_PATH = ROOT / "data" / "icd_reference" / "dx_aliases_ru.json"

_TOKEN_BOUND = r"(?<![а-яёa-z0-9]){stem}(?![а-яёa-z0-9])"


def _stem_in_text(low: str, stem: str) -> bool:
    stem = (stem or "").strip().lower()
    if not stem or not low:
        return False
    return bool(re.search(_TOKEN_BOUND.format(stem=re.escape(stem)), low, re.IGNORECASE))


@lru_cache(maxsize=1)
def _load_alias_file() -> dict[str, Any]:
    if not _ALIASES_PATH.is_file():
        return {"abbreviations": [], "word_expansions": []}
    try:
        raw = json.loads(_ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"abbreviations": [], "word_expansions": []}
    if not isinstance(raw, dict):
        return {"abbreviations": [], "word_expansions": []}
    return raw


def _strip_title_code(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"^[A-TV-Z]\d{2}(?:\.\d{1,4})?\s*[-–—:]?\s*", "", t, flags=re.I)
    return t.strip()


def _apply_word_expansions(text: str, expansions: list[dict[str, Any]]) -> str:
    out = text or ""
    # длинные alias раньше (хрон. до хр.)
    rows = sorted(
        (r for r in expansions if isinstance(r, dict) and r.get("alias") and r.get("expand")),
        key=lambda r: len(str(r["alias"])),
        reverse=True,
    )
    for row in rows:
        alias = str(row["alias"])
        expand = str(row["expand"])
        # сокращения с точкой: хр. / остр.
        if alias.endswith("."):
            pat = re.compile(
                r"(?<![а-яёa-z0-9])" + re.escape(alias[:-1]) + r"\.?(?![а-яёa-z0-9])",
                re.IGNORECASE,
            )
            out = pat.sub(expand, out)
        else:
            pat = re.compile(
                _TOKEN_BOUND.format(stem=re.escape(alias)),
                re.IGNORECASE,
            )
            out = pat.sub(expand, out)
    return out


def expand(diag_text: str) -> dict[str, Any]:
    """Расширить формулировку Dx алиасами и seed из diagnosis_icd."""
    original = (diag_text or "").strip()
    empty = {
        "original": original,
        "normalized": original,
        "expanded_phrases": [],
        "seed_codes": [],
        "match_method": None,
        "match_query": original,
    }
    if not original:
        return empty

    data = _load_alias_file()
    word_exp = list(data.get("word_expansions") or [])
    abbrevs = list(data.get("abbreviations") or [])

    normalized = _apply_word_expansions(original, word_exp)
    low = normalized.lower().replace("ё", "е")

    phrases: list[str] = []
    seed_codes: list[str] = []
    methods: list[str] = []

    # JSON-аббревиатуры (длинные раньше)
    abbrev_rows = sorted(
        (r for r in abbrevs if isinstance(r, dict) and r.get("alias")),
        key=lambda r: len(str(r["alias"])),
        reverse=True,
    )
    for row in abbrev_rows:
        alias = str(row["alias"]).strip().lower().replace("ё", "е")
        if not _stem_in_text(low, alias):
            continue
        expand_phrase = str(row.get("expand") or "").strip()
        code = str(row.get("seed_code") or "").strip().upper()
        if expand_phrase and expand_phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(expand_phrase)
        if code and code not in seed_codes:
            seed_codes.append(code)
        methods.append("alias_json")

    # Seed нозология→код из diagnosis_icd
    try:
        from clinical_knowledge.diagnosis_icd import _DIAGNOSIS_ICD_SEED, lookup_disease_icd

        for stem, code in _DIAGNOSIS_ICD_SEED:
            if not _stem_in_text(low, stem):
                continue
            code_n = str(code).strip().upper()
            if code_n and code_n not in seed_codes:
                seed_codes.append(code_n)
            methods.append("alias_seed")
            try:
                import icd_mkb

                title = icd_mkb.ru_title(code_n)
                phrase = _strip_title_code(title or "")
                if phrase and phrase.lower() not in {p.lower() for p in phrases}:
                    phrases.append(phrase)
            except Exception:  # noqa: BLE001
                pass
        # на случай если seed дал коды без stem-loop (lookup тот же)
        for code in lookup_disease_icd(normalized):
            code_n = str(code).strip().upper()
            if code_n and code_n not in seed_codes:
                seed_codes.append(code_n)
    except Exception:  # noqa: BLE001
        pass

    # title_ru для seed_codes без phrase
    if seed_codes:
        try:
            import icd_mkb

            for code in seed_codes:
                title = icd_mkb.ru_title(code)
                phrase = _strip_title_code(title or "")
                if phrase and phrase.lower() not in {p.lower() for p in phrases}:
                    phrases.append(phrase)
        except Exception:  # noqa: BLE001
            pass

    parts = [normalized]
    for p in phrases:
        if p and p.lower() not in normalized.lower():
            parts.append(p)
    match_query = " ".join(parts).strip()
    method = None
    if methods:
        method = "alias_seed" if "alias_seed" in methods else "alias_json"
    elif normalized != original:
        method = "word_expand"

    return {
        "original": original,
        "normalized": normalized,
        "expanded_phrases": phrases,
        "seed_codes": seed_codes,
        "match_method": method,
        "match_query": match_query,
    }


def match_query(diag_text: str) -> str:
    """Текст для suggest / name_match (original + expansions)."""
    return str(expand(diag_text).get("match_query") or diag_text or "").strip()
