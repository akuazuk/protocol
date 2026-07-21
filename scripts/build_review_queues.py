"""Очереди доводки карточек протоколов (P3).

Сканирует опубликованные Summary Cards и формирует:
  1. data/protocol_summaries/reextract_regex_queue.json - id протоколов с регекс-качеством
     (`extraction_status` in {auto_extracted, draft}) для LLM-переизвлечения (форс-режим);
  2. data/protocol_summaries/methodist_review_queue.(json|md) - протоколы с
     review_status=needs_review для ручного ревью методистом (регекс-карточки - в приоритете).

Запуск:
  python3 scripts/build_review_queues.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "protocol_summaries"


def main() -> None:
    from clinical_knowledge.protocol_summary.loader import load_protocol_summaries

    summaries = load_protocol_summaries(usable_only=False)
    regex_queue: list[dict] = []
    review: list[dict] = []
    status_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    weak = {"auto_extracted", "draft"}

    for s in summaries:
        es = str(getattr(s, "extraction_status", "") or "")
        rs = str(getattr(s, "review_status", "") or "")
        status_counts[es] = status_counts.get(es, 0) + 1
        review_counts[rs] = review_counts.get(rs, 0) + 1
        row = {
            "protocol_id": s.protocol_id,
            "title": s.source.title or "",
            "local_path": s.source.local_path or "",
            "extraction_status": es,
            "review_status": rs,
        }
        if es in weak:
            regex_queue.append(row)
        if rs == "needs_review":
            review.append(row)

    # Регекс-карточки - в приоритете ревью.
    review.sort(key=lambda r: (0 if r["extraction_status"] in weak else 1, r["protocol_id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reextract_regex_queue.json").write_text(
        json.dumps([r["protocol_id"] for r in regex_queue], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "methodist_review_queue.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = ["# Очереди доводки карточек протоколов", ""]
    md.append(f"Всего сводок: {len(summaries)}")
    md.append(f"extraction_status: {json.dumps(status_counts, ensure_ascii=False)}")
    md.append(f"review_status: {json.dumps(review_counts, ensure_ascii=False)}")
    md.append("")
    md.append(f"## Регекс-качество -> LLM-переизвлечение (форс): {len(regex_queue)}")
    for r in regex_queue:
        md.append(f"- `{r['protocol_id']}` ({r['extraction_status']}) - {r['title']}")
    md.append("")
    md.append(f"## Ручное ревью методиста (review_status=needs_review): {len(review)}")
    md.append("_Регекс-карточки идут первыми - у них ниже качество._")
    for r in review[:80]:
        md.append(f"- `{r['protocol_id']}` ({r['extraction_status']}) - {r['title']}")
    if len(review) > 80:
        md.append(f"- … ещё {len(review) - 80}")
    (OUT_DIR / "methodist_review_queue.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"summaries: {len(summaries)}")
    print(f"extraction_status: {status_counts}")
    print(f"review_status: {review_counts}")
    print(f"regex/draft queue: {len(regex_queue)} -> reextract_regex_queue.json")
    print(f"methodist review: {len(review)} -> methodist_review_queue.(json|md)")


if __name__ == "__main__":
    main()
