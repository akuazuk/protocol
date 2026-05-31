#!/usr/bin/env python3
"""Обогащение карточек протоколов кодами МКБ-10 из текста КП.

Карточки реестра (output/registry/protocol_cards.jsonl) у многих PDF имеют
пустые icd10_all/icd10_primary, из-за чего детерминированный подбор протокола
«диагноз -> код -> КП» не срабатывает. Скрипт агрегирует коды МКБ по каждому
PDF из чанков (output/chunks/chunks.jsonl, поле icd10_codes) и проставляет их
карточкам этого PDF.

Аддитивно: существующие коды не удаляются, только дополняются. Перед записью
делается резервная копия protocol_cards.jsonl.bak.

Запуск:
  python -m scripts.enrich_cards_icd            # применить (с бэкапом)
  python -m scripts.enrich_cards_icd --dry-run  # только статистика
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"
CARDS = ROOT / "output" / "registry" / "protocol_cards.jsonl"

# Сколько кодов брать в icd10_primary (по частоте упоминаний в PDF).
PRIMARY_LIMIT = 12


def _norm_path(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _norm_code(code: str) -> str:
    return (code or "").upper().strip().replace(" ", "")


def build_path_icd_map() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Возвращает (path -> все коды отсортированные, path -> коды по частоте desc)."""
    freq: dict[str, Counter] = defaultdict(Counter)
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            sp = _norm_path(c.get("source_path") or "")
            if not sp:
                continue
            for code in c.get("icd10_codes") or []:
                cc = _norm_code(str(code))
                if len(cc) >= 3:
                    freq[sp][cc] += 1
    all_codes = {sp: sorted(cnt) for sp, cnt in freq.items()}
    by_freq = {
        sp: [code for code, _ in cnt.most_common()]
        for sp, cnt in freq.items()
    }
    return all_codes, by_freq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Только статистика, без записи")
    args = parser.parse_args()

    if not CHUNKS.is_file() or not CARDS.is_file():
        print(json.dumps({"error": "chunks or cards file missing"}, ensure_ascii=False))
        return 1

    all_codes, by_freq = build_path_icd_map()
    print(f"PDF с кодами в чанках: {len(all_codes)}")

    cards: list[dict] = []
    with CARDS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cards.append(json.loads(line))

    total = len(cards)
    had_icd_before = sum(1 for c in cards if c.get("icd10_all") or c.get("icd10_primary"))
    enriched = 0
    added_all_total = 0

    for card in cards:
        sp = _norm_path(str(card.get("source_path") or ""))
        path_codes = all_codes.get(sp)
        if not path_codes:
            continue
        existing_all = {_norm_code(str(x)) for x in (card.get("icd10_all") or []) if x}
        merged = existing_all | set(path_codes)
        if merged != existing_all:
            added_all_total += len(merged) - len(existing_all)
        card["icd10_all"] = sorted(merged)
        if not card.get("icd10_primary"):
            card["icd10_primary"] = (by_freq.get(sp) or [])[:PRIMARY_LIMIT]
        enriched += 1

    had_icd_after = sum(1 for c in cards if c.get("icd10_all") or c.get("icd10_primary"))

    summary = {
        "cards_total": total,
        "cards_with_icd_before": had_icd_before,
        "cards_with_icd_after": had_icd_after,
        "cards_enriched": enriched,
        "icd_codes_added": added_all_total,
        "pdfs_with_codes": len(all_codes),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    backup = CARDS.with_suffix(".jsonl.bak")
    backup.write_text(CARDS.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = CARDS.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    os.replace(tmp, CARDS)
    print(f"Записано {total} карточек. Бэкап: {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
