#!/usr/bin/env python3
"""Пересчёт архивных снимков анализа КЗ и сравнение с сохранёнными метриками.

Использование:
  python scripts/replay_consult_archive.py                    # все из manifest
  python scripts/replay_consult_archive.py --limit 20         # последние 20
  python scripts/replay_consult_archive.py --fixtures path.jsonl  # фикстуры из git

Если в clients_consult/ есть файл с тем же basename, что в снимке — пересчитывает
структурный анализ и сравнивает баллы/статус/МКБ. Иначе — только сводка архива.

Полезно после изменений парсера/матчера: подтянуть manifest с prod, положить в
fixtures/consult_replay.jsonl и запустить локально перед коммитом.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.analysis_archive import archive_dir, load_snapshots  # noqa: E402
from clinical_knowledge.consult_analysis import analyze_consultation_text  # noqa: E402


def _load_text(basename: str) -> str | None:
    if not basename:
        return None
    for folder in ("clients_consult", "tests/fixtures/consult"):
        p = ROOT / folder / basename
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                try:
                    import fitz
                    return "\n".join(page.get_text() for page in fitz.open(p))
                except Exception:
                    return None
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _compare(saved: dict, fresh: dict) -> list[str]:
    diffs: list[str] = []
    keys = (
        ("structured_overall_status", "overall_status"),
        ("structured_overall_score", "overall_score"),
    )
    for old_k, new_k in keys:
        old_v = saved.get(old_k)
        new_v = fresh.get(new_k)
        if old_v != new_v:
            if isinstance(old_v, float) and isinstance(new_v, (int, float)):
                if abs(float(old_v) - float(new_v)) < 0.5:
                    continue
            diffs.append(f"{old_k}: {old_v!r} → {new_v!r}")
    old_icd = saved.get("icd_codes") or []
    new_icd = []
    for d in (fresh.get("diagnosis_assessments") or []):
        c = d.get("icd10_code")
        if c:
            new_icd.append(c)
    if old_icd and new_icd and set(old_icd[:3]) != set(new_icd[:3]):
        diffs.append(f"icd_codes: {old_icd[:3]!r} → {new_icd[:3]!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay consult analysis archive")
    ap.add_argument("--limit", type=int, default=0, help="Max snapshots (0 = all)")
    ap.add_argument("--fixtures", type=str, default="", help="JSONL file instead of manifest")
    args = ap.parse_args()

    if args.fixtures:
        path = Path(args.fixtures)
        snaps = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    else:
        limit = args.limit or None
        snaps = load_snapshots(limit=limit)

    if not snaps:
        print(f"No snapshots in {archive_dir() / 'manifest.jsonl'}")
        return 1

    ok = 0
    skipped = 0
    failed = 0
    print(f"Replay {len(snaps)} snapshot(s)\n")

    for i, snap in enumerate(snaps, 1):
        base = snap.get("source_basename") or ""
        text = _load_text(base)
        label = base or snap.get("text_hash", "")[:12]
        if not text:
            skipped += 1
            print(f"[{i}] SKIP {label} — файл не найден локально")
            continue
        res = analyze_consultation_text(text, with_markdown=False)
        comp = res.get("compliance") or {}
        diffs = _compare(snap, comp)
        if diffs:
            failed += 1
            print(f"[{i}] DIFF {label}")
            for d in diffs:
                print(f"      {d}")
        else:
            ok += 1
            score = comp.get("overall_score")
            status = comp.get("overall_status")
            print(f"[{i}] OK   {label} — {status} {score}%")

    print(f"\nSummary: ok={ok} diff={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
