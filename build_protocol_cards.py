#!/usr/bin/env python3
"""Строит output/registry/protocol_cards.jsonl и data/gastro_mvp/protocol_registry.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus_pipeline.protocol_cards import build_all_protocol_cards, write_protocol_cards_jsonl

GASTRO_SLUG = "gastroenterologiya"


def main() -> None:
    cards = build_all_protocol_cards(ROOT)
    out_all = ROOT / "output" / "registry" / "protocol_cards.jsonl"
    write_protocol_cards_jsonl(cards, out_all)

    gastro = [c for c in cards if c.get("specialty_slug") == GASTRO_SLUG]
    gastro_dir = ROOT / "data" / "gastro_mvp"
    gastro_dir.mkdir(parents=True, exist_ok=True)
    gastro_out = gastro_dir / "protocol_registry.jsonl"
    write_protocol_cards_jsonl(gastro, gastro_out)

    summary = {
        "total_cards": len(cards),
        "gastro_cards": len(gastro),
        "output": str(out_all.relative_to(ROOT)),
        "gastro_output": str(gastro_out.relative_to(ROOT)),
    }
    (gastro_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OK: {len(cards)} карточек → {out_all}")
    print(f"OK: {len(gastro)} гастро-карточек → {gastro_out}")


if __name__ == "__main__":
    main()
