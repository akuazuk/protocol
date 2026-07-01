#!/usr/bin/env python3
"""Сквозной аудит чистоты выдержек L2 по всем протоколам.

Для каждой usable-сводки строит блок-выдержки (как build_evidence_pack) и
проверяет каждое предложение на остаточный мусор (колонтитулы, № постановлений,
преамбулы, оргтекст, обрезки). Печатает сводную статистику и примеры мусора.

Usage:
    python3 scripts/audit_summary_excerpts.py [--limit N] [--show N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.consult_evidence_pack import (  # noqa: E402
    EVIDENCE_BLOCK_IDS,
    _emit_condition_excerpts,
)
from clinical_knowledge.protocol_summary.loader import (  # noqa: E402
    load_protocol_summaries,
)

# Паттерны "мусора", который НЕ должен попадать в клиническую выдержку.
GARBAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("page_marker", re.compile(r"стр\.?\s*\d", re.I)),
    ("order_number", re.compile(r"№\s*\d")),
    ("portal", re.compile(r"национальн\w+\s+правов|интернет-портал", re.I)),
    ("postanovlenie", re.compile(r"постановлени|пастанова|приказ\w*\s+министерств", re.I)),
    ("prilozhenie", re.compile(r"приложени\w*\s*\d|к\s+клиническому\s+протоколу", re.I)),
    ("minister_sig", re.compile(r"министр\s+[а-яё]\.\s?[а-яё]\.", re.I)),
    ("scope_clause", re.compile(r"устанавливает\s+общие\s+требовани|настоящий\s+клиническ", re.I)),
    ("org_routing", re.compile(r"направля\w+\s+пациент|порядок\s+направлени|выполняют\s+врачи|определяется\s+министерством", re.I)),
    ("date_only", re.compile(r"^\s*\d{1,2}[.\s]\d{2}[.\s]\d{4}", re.I)),
    ("zaklyuchenie_blank", re.compile(r"заключение:\s*_{3,}", re.I)),
    ("truncated_tail", re.compile(r"\b(?:наблюден|обследован|консультаци|вмешательств|показани)$", re.I)),
]


# Дополнительные "подозрительные" эвристики (шире, чем фильтр), чтобы искать
# новый мусор, который фильтр ещё не знает.
SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("title_reference", re.compile(r"^«[^»]{20,}»\.?$")),
    ("all_caps_header", re.compile(r"^[А-ЯЁ\s]{8,}$")),
    ("in_accordance", re.compile(r"в\s+соответствии\s+с|утвержд[её]нн?", re.I)),
    ("determined_by", re.compile(r"определяется|осуществляется\s+в", re.I)),
    ("who_performs", re.compile(r"выполня\w+\s+врач|проводится\s+врач", re.I)),
    ("etc_terms", re.compile(r"а\s+также\s+следующ|термин\w+\s+и\s+их\s+определени", re.I)),
    ("mid_sentence_start", re.compile(r"^[а-яё]{1,3}\s")),
]


def find_suspicious(text: str) -> list[str]:
    t = (text or "").strip()
    return [name for name, pat in SUSPICIOUS_PATTERNS if pat.search(t)]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def find_garbage(text: str) -> list[str]:
    hits: list[str] = []
    stripped = (text or "").rstrip(".!?;:,) ")
    for name, pat in GARBAGE_PATTERNS:
        target = stripped if name == "truncated_tail" else text
        if pat.search(target or ""):
            hits.append(name)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ограничить число протоколов")
    ap.add_argument("--show", type=int, default=25, help="сколько примеров мусора печатать")
    ap.add_argument("--samples", type=int, default=8, help="сколько примеров выдержек на блок")
    ap.add_argument("--json", type=str, default=None, help="сохранить отчёт в JSON")
    args = ap.parse_args()

    summaries = load_protocol_summaries(usable_only=True)
    if args.limit:
        summaries = summaries[: args.limit]

    total = len(summaries)
    with_excerpt = 0
    garbage_protocols: set[str] = set()
    garbage_examples: list[dict] = []
    block_counts = {b: 0 for b in EVIDENCE_BLOCK_IDS}
    total_excerpts = 0
    total_sentences = 0
    garbage_sentences = 0
    garbage_by_type: dict[str, int] = {}
    suspicious_examples: list[dict] = []
    suspicious_by_type: dict[str, int] = {}
    samples_by_block: dict[str, list[str]] = {b: [] for b in EVIDENCE_BLOCK_IDS}

    for s in summaries:
        out = {b: [] for b in EVIDENCE_BLOCK_IDS}
        for cond in s.conditions:
            _emit_condition_excerpts(out, src_path="x", cond=cond, max_per_block=6)
        n = sum(len(v) for v in out.values())
        if n:
            with_excerpt += 1
        for b in EVIDENCE_BLOCK_IDS:
            if out[b]:
                block_counts[b] += 1
            for item in out[b]:
                total_excerpts += 1
                if len(samples_by_block[b]) < args.samples:
                    samples_by_block[b].append(item["excerpt"][:200])
                susp = find_suspicious(item["excerpt"])
                if susp:
                    for h in susp:
                        suspicious_by_type[h] = suspicious_by_type.get(h, 0) + 1
                    if len(suspicious_examples) < 500:
                        suspicious_examples.append(
                            {"protocol_id": s.protocol_id, "block": b,
                             "types": susp, "excerpt": item["excerpt"][:160]}
                        )
                for sent in split_sentences(item["excerpt"]):
                    total_sentences += 1
                    hits = find_garbage(sent)
                    if hits:
                        garbage_sentences += 1
                        garbage_protocols.add(s.protocol_id)
                        for h in hits:
                            garbage_by_type[h] = garbage_by_type.get(h, 0) + 1
                        if len(garbage_examples) < 500:
                            garbage_examples.append(
                                {
                                    "protocol_id": s.protocol_id,
                                    "block": b,
                                    "types": hits,
                                    "sentence": sent[:160],
                                }
                            )

    print(f"protocols usable={total}  with>=1 excerpt={with_excerpt} ({100*with_excerpt/total:.1f}%)")
    print(f"per-block protocols: {block_counts}")
    print(f"total excerpts={total_excerpts}  sentences={total_sentences}")
    print(
        f"GARBAGE: sentences={garbage_sentences} "
        f"({100*garbage_sentences/max(1,total_sentences):.2f}%)  "
        f"protocols={len(garbage_protocols)}"
    )
    print(f"garbage by type: {dict(sorted(garbage_by_type.items(), key=lambda x: -x[1]))}")
    print(f"--- garbage examples (first {args.show}) ---")
    for ex in garbage_examples[: args.show]:
        print(f"[{ex['block']}] {ex['types']} :: {ex['sentence']}")
    print(
        f"SUSPICIOUS: hits={len(suspicious_examples)} "
        f"by type={dict(sorted(suspicious_by_type.items(), key=lambda x: -x[1]))}"
    )
    for ex in suspicious_examples[: args.show]:
        print(f"  ?[{ex['block']}] {ex['types']} :: {ex['excerpt']}")
    print("--- excerpt samples per block ---")
    for b in EVIDENCE_BLOCK_IDS:
        print(f"### {b}")
        for smp in samples_by_block[b]:
            print(f"    {smp}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "total": total,
                    "with_excerpt": with_excerpt,
                    "block_counts": block_counts,
                    "total_excerpts": total_excerpts,
                    "total_sentences": total_sentences,
                    "garbage_sentences": garbage_sentences,
                    "garbage_protocols": sorted(garbage_protocols),
                    "garbage_by_type": garbage_by_type,
                    "examples": garbage_examples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
