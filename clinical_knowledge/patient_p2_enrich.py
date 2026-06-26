"""P2 enrichment: plain-language narrative без LLM (rule-based)."""
from __future__ import annotations

from typing import Any


def enrich_patient_report_p2(report: dict[str, Any]) -> dict[str, Any]:
    """Добавляет развёрнутые пояснения для tier P2."""
    out = dict(report)
    narratives: list[dict[str, str]] = []
    for b in report.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        status = b.get("status")
        if status == "ok":
            continue
        title = str(b.get("title") or "Раздел")
        why = str(b.get("why_ru") or b.get("summary_ru") or "").strip()
        excerpt = str(b.get("protocol_excerpt") or "").strip()
        parts = [f"В разделе «{title}» есть повод уточнить детали у врача."]
        if why:
            parts.append(why)
        if excerpt:
            parts.append(f"По протоколу Минздрава: {excerpt[:280]}")
        narratives.append({"block_id": str(b.get("id") or ""), "title": title, "text_ru": " ".join(parts)})
    out["plain_narratives"] = narratives[:8]
    out["review_tier_product"] = "P2"
    steps = list(out.get("next_steps_ru") or [])
    if narratives and "Прочитайте пояснения к блокам с замечаниями." not in steps:
        steps.insert(1, "Прочитайте пояснения к блокам с замечаниями.")
    out["next_steps_ru"] = steps
    return out
