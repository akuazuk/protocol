#!/usr/bin/env python3
"""Собрать golden_protocol_pairs.jsonl из retrieval_fix и analysis_review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iter_events(feedback_dir: Path) -> list[dict]:
    events: list[dict] = []
    if not feedback_dir.is_dir():
        return events
    for path in sorted(feedback_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                events.append(row)
    return events


def build_pairs(feedback_dir: Path) -> list[dict]:
    events = _iter_events(feedback_dir)
    kz_by_id = {
        str(e.get("analysis_id")): e
        for e in events
        if e.get("event_type") == "kz_analysis" and e.get("analysis_id")
    }
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(*, source: str, ev: dict, chosen: str, rejected: str = "", query: str = "") -> None:
        chosen = (chosen or "").strip()
        if not chosen:
            return
        th = str(ev.get("text_hash") or "")
        aid = str(ev.get("analysis_id") or "")
        kz = kz_by_id.get(aid) or {}
        diag_icd = list(kz.get("icd_codes") or kz.get("icd10") or [])[:6]
        if not diag_icd:
            diag_icd = []
        rubric = str(kz.get("rubric") or "")
        key = (th or aid, chosen)
        if key in seen:
            return
        seen.add(key)
        pairs.append(
            {
                "text_hash": th,
                "analysis_id": aid,
                "source": source,
                "query": (query or ev.get("query") or "")[:500],
                "diagnosis_icd": diag_icd,
                "rubric": rubric,
                "rejected_path": (rejected or "").strip(),
                "chosen_path": chosen,
                "ts": ev.get("ts"),
            }
        )

    for ev in events:
        et = ev.get("event_type")
        if et == "retrieval_fix":
            add_pair(
                source="retrieval_fix",
                ev=ev,
                chosen=str(ev.get("chosen_path") or ""),
                rejected=str(ev.get("rejected_path") or ""),
                query=str(ev.get("query") or ""),
            )
        elif et == "analysis_review":
            rf = ev.get("retrieval_fix")
            if isinstance(rf, dict):
                add_pair(
                    source="analysis_review",
                    ev=ev,
                    chosen=str(rf.get("chosen_path") or ""),
                    rejected=str(rf.get("rejected_path") or ""),
                    query=str(rf.get("query") or ""),
                )

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--feedback-dir",
        type=Path,
        default=ROOT / "data" / "ml" / "feedback",
        help="Каталог JSONL feedback",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "ml" / "datasets" / "golden_protocol_pairs.jsonl",
    )
    args = ap.parse_args()
    pairs = build_pairs(args.feedback_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs → {args.out}")


if __name__ == "__main__":
    main()
