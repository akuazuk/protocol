#!/usr/bin/env python3
"""Скачать базу лекарственных взаимодействий DDInter 2.0 и собрать единый индекс.

Источник: http(s)://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_<C>.csv
Категории <C> - анатомические группы ATC (A,B,D,G,H,J,L,M,N,P,R,S,V).
CSV-структура: DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level (Major/Moderate/Minor/Unknown).

На выход:
  data/drug_safety/ddinter/ddinter_downloads_code_<C>.csv  - сырьё (для аудита)
  data/drug_safety/ddinter_pairs.json                       - {"pairs": {"a||b": level}, ...}

Взаимодействия ищутся по канону INN (нижний регистр). Русские названия из КЗ
приводятся к INN в clinical_knowledge/drug_normalizer.py (шаг 4.3 ТЗ).

Usage:
  python3 scripts/fetch_ddinter.py            # скачать всё
  python3 scripts/fetch_ddinter.py --offline  # только пересобрать из уже скачанных CSV
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import ssl
import sys
import urllib.request
from pathlib import Path

try:
    import certifi

    _SSL_CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = None

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "drug_safety" / "ddinter"
PAIRS_JSON = ROOT / "data" / "drug_safety" / "ddinter_pairs.json"

CATEGORIES = ["A", "B", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]
BASE = "https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{c}.csv"

# Уровни DDInter -> внутренняя тяжесть (для risk-gate).
LEVEL_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}


def _norm(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _pair_key(a: str, b: str) -> str:
    x, y = sorted((_norm(a), _norm(b)))
    return f"{x}||{y}"


def download(timeout: float = 60.0) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in CATEGORIES:
        url = BASE.format(c=c)
        dst = OUT_DIR / f"ddinter_downloads_code_{c}.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "protocol-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                data = r.read()
            dst.write_bytes(data)
            print(f"  {c}: {len(data)} bytes -> {dst.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: FAIL {type(e).__name__}: {e}", file=sys.stderr)


def build() -> dict:
    pairs: dict[str, str] = {}
    drugs: set[str] = set()
    rows_total = 0
    for c in CATEGORIES:
        src = OUT_DIR / f"ddinter_downloads_code_{c}.csv"
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            a = row.get("Drug_A") or ""
            b = row.get("Drug_B") or ""
            lvl = (row.get("Level") or "Unknown").strip().title()
            if not a or not b:
                continue
            rows_total += 1
            drugs.add(_norm(a))
            drugs.add(_norm(b))
            key = _pair_key(a, b)
            # при дублях оставляем максимальную тяжесть
            if key in pairs and LEVEL_RANK.get(pairs[key], 0) >= LEVEL_RANK.get(lvl, 0):
                continue
            pairs[key] = lvl
    out = {
        "source": "DDInter 2.0 (ddinter2.scbdd.com)",
        "level_rank": LEVEL_RANK,
        "n_rows_raw": rows_total,
        "n_pairs": len(pairs),
        "n_drugs": len(drugs),
        "pairs": pairs,
        "drugs": sorted(drugs),
    }
    PAIRS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PAIRS_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="не качать, собрать из CSV в data/drug_safety/ddinter")
    args = ap.parse_args()
    if not args.offline:
        print("Downloading DDInter category CSVs...")
        download()
    print("Building combined index...")
    out = build()
    print(f"Done: pairs={out['n_pairs']} drugs={out['n_drugs']} (raw rows={out['n_rows_raw']}) -> {PAIRS_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
