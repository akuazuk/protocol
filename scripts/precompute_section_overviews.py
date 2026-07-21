"""Предрасчёт ИИ-обзоров разделов протоколов (P5).

Для каждого протокола из очереди генерирует «суть раздела» (Gemini) по разделам
навигатора и кладёт в кэш `data/ml/section_overviews/{protocol_id}.json`, который
читает `/api/protocol-brief`. Локально из РБ Gemini geo-blocked - запускать на Render
Web Shell (как батч LLM-переизвлечения).

Использование:
  python3 scripts/precompute_section_overviews.py --top 40
  python3 scripts/precompute_section_overviews.py --paths path1,path2

Без ключа Gemini скрипт ничего не пишет и сообщает об этом.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SECTION_QUERIES: dict[str, str] = {
    "diagnosis": "критерии диагноза и классификация",
    "exams": "какие обследования назначить",
    "treatment": "схема лечения и препараты с дозами",
    "red_flags": "красные флаги и когда госпитализировать",
    "follow_up": "наблюдение и маршрутизация пациента",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="сколько протоколов взять (по каталогу)")
    ap.add_argument("--paths", type=str, default="", help="список local_path через запятую")
    args = ap.parse_args()

    from clinical_knowledge.protocol_semantic_search import build_protocol_overview
    from clinical_knowledge.section_overview_cache import save_section_overviews

    try:
        from clinical_knowledge.gemini_lite import gemini_available
        if not gemini_available():
            print("Gemini недоступен (ключ/гео). Обзоры не сгенерированы.")
            return
    except Exception:
        print("Gemini-модуль недоступен. Обзоры не сгенерированы.")
        return

    from clinical_knowledge.protocol_summary.loader import load_protocol_summaries

    summaries = load_protocol_summaries(usable_only=False)
    by_path = {s.source.local_path: s for s in summaries if s.source.local_path}

    if args.paths.strip():
        paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    else:
        paths = list(by_path.keys())
        if args.top > 0:
            paths = paths[: args.top]

    done = 0
    for path in paths:
        summ = by_path.get(path)
        pid = summ.protocol_id if summ else path
        title = summ.source.title if summ else ""
        sections: dict[str, dict] = {}
        for sec_id, q in _SECTION_QUERIES.items():
            try:
                ov = build_protocol_overview(path, q, title=title or "")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {pid}/{sec_id}: {exc}")
                continue
            if ov and ov.get("ok") and (ov.get("summary") or ov.get("points")):
                sections[sec_id] = {"summary": ov.get("summary", ""), "points": ov.get("points", [])}
        if sections:
            save_section_overviews(pid, sections)
            done += 1
            print(f"  ok {pid}: {list(sections.keys())}")

    print(f"Готово: обзоры для {done}/{len(paths)} протоколов.")


if __name__ == "__main__":
    main()
