"""Проба: сравнить старый source-view (сырые чанки) и новую сводку protocol_brief.

Метрики: число точек, дублей и обрывков (эвристики). Запуск:
  python3 scripts/probe_protocol_brief.py [<catalog_path>]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RICH = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"

DEFAULT_PATH = (
    "minzdrav_protocols/pulmonologiya-ftiziatriya/"
    "КП_Диагностика_и_лечение_взр_население_с_бронхиальной_астмой_пост_МЗ_2024_84.pdf"
)


def _rich_for_path(path: str) -> list[dict]:
    norm = path.replace("\\", "/").strip()
    base = norm.split("/")[-1]
    rows: list[dict] = []
    with RICH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = str(ch.get("source_path") or ch.get("path") or "").replace("\\", "/")
            if sp == norm or sp.endswith(base):
                rows.append(ch)
    return rows


def _looks_fragment(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 24:
        return True
    if t.endswith(("."[0],)) and t[:-1].strip().split(" ")[-1].lower() in ("и", "с", "по", "для", "далее"):
        return True
    return t.count("(") > t.count(")")


def _count_dups(texts: list[str]) -> int:
    seen: set[str] = set()
    dups = 0
    for t in texts:
        key = re.sub(r"[^а-яa-z0-9 ]+", " ", (t or "").lower().replace("ё", "е"))
        key = re.sub(r"\s+", " ", key).strip()[:80]
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    rich = _rich_for_path(path)
    print(f"path: {path}\nrich chunks loaded: {len(rich)}\n")

    from clinical_knowledge.protocol_source_view import prepare_protocol_source_view
    from clinical_knowledge.protocol_summary.source_text import _doc_from_rich_chunks

    old_texts: list[str] = []
    doc = _doc_from_rich_chunks(rich, path=path)
    if doc is not None:
        view = prepare_protocol_source_view(doc)
        for grp in view.get("toc") or []:
            for item in (view.get("sections") or {}).get(grp["id"], []) if isinstance(view.get("sections"), dict) else []:
                old_texts.append(item.get("lead") or "")
    # универсальный обход: собрать все lead из view
    if not old_texts and doc is not None:
        view = prepare_protocol_source_view(doc)
        def _walk(o):
            if isinstance(o, dict):
                if "lead" in o and isinstance(o["lead"], str):
                    old_texts.append(o["lead"])
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for v in o:
                    _walk(v)
        _walk(view)

    from clinical_knowledge.protocol_brief import build_protocol_brief

    brief = build_protocol_brief(
        path,
        query="бронхиальная астма",
        rich_chunks=rich or None,
    )
    new_texts = [p["text"] for s in brief.get("sections", []) for p in s["points"]]

    print("== OLD source-view (leads) ==")
    print(f"  items:      {len(old_texts)}")
    print(f"  duplicates: {_count_dups(old_texts)}")
    print(f"  fragments:  {sum(1 for t in old_texts if _looks_fragment(t))}")
    print()
    print("== NEW protocol_brief ==")
    print(f"  source:     {brief.get('source')}")
    print(f"  condition:  {(brief.get('condition') or {}).get('display_label')}")
    print(f"  sections:   {[ (s['id'], s['count']) for s in brief.get('sections', []) ]}")
    print(f"  points:     {len(new_texts)}")
    print(f"  duplicates: {_count_dups(new_texts)}")
    print(f"  fragments:  {sum(1 for t in new_texts if _looks_fragment(t))}")
    print(f"  entities.drugs: {brief.get('entities', {}).get('drugs')}")
    print(f"  entities.exams: {brief.get('entities', {}).get('exams')}")
    print("\n-- sample NEW points --")
    for s in brief.get("sections", []):
        print(f"[{s['label']}]")
        for p in s["points"]:
            pg = f" (стр. {p['page_start']})" if p.get("page_start") else ""
            vf = " ✓цитата" if p.get("verified") else ""
            print(f"   - {p['text']}{pg}{vf}")


if __name__ == "__main__":
    main()
