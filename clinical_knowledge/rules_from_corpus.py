"""Извлечение правил из размеченного корпуса (chunks.jsonl) — эвристики по тексту КП."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# (condition_id, regex для поиска блока «формулировка диагноза …»)
CONDITION_DIAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gerd", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+гэрб", re.I)),
    ("peptic_ulcer", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+гастродуоденальн", re.I)),
    ("gastritis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+хроническ", re.I)),
    ("functional_dyspepsia", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+функциональн", re.I)),
]

CONDITION_CRIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gerd", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+гэрб", re.I)),
    ("peptic_ulcer", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+гастродуоденальн", re.I)),
    ("gastritis", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+хроническ", re.I)),
]

RE_INCLUDES_BULLETS = re.compile(
    r"включает\s*:?\s*(.+?)(?=пример|^\d+\.|диагностическ|национальн|\Z)",
    re.I | re.S | re.M,
)
RE_GERD_SYMPTOM_CRIT = re.compile(
    r"изжога\s+или\s+кислая\s+регургитация[^.\n]{0,120}",
    re.I,
)


def _parse_required_components(block: str) -> list[str]:
    """Компоненты формулировки диагноза из текста после «включает»."""
    components: list[str] = []
    low = block.lower()
    mapping = [
        ("нозолог", "нозология"),
        ("клиническ", "клиническая форма"),
        ("степен", "степень тяжести"),
        ("фаз", "фаза"),
        ("осложнен", "осложнения"),
        ("локализац", "локализация"),
        ("нр-инфекц", "H.pylori"),
        ("этиолог", "этиология"),
    ]
    for needle, label in mapping:
        if needle in low and label not in components:
            components.append(label)
    if not components:
        for line in block.split("\n"):
            line = line.strip().strip(";,")
            if 3 < len(line) < 80 and not line.startswith("пример"):
                components.append(line[:80])
            if len(components) >= 6:
                break
    return components[:8]


def _rule_source(chunk: dict[str, Any], protocol_id: str) -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "doc_id": chunk.get("doc_id"),
        "page_from": chunk.get("page_from"),
        "page_to": chunk.get("page_to"),
        "section": chunk.get("section_title"),
        "section_path": chunk.get("section_path"),
        "quote_chars": min(400, len(chunk.get("text") or "")),
    }


def extract_rules_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    protocol_id: str = "gastro_kp185",
) -> dict[str, list[dict[str, Any]]]:
    """Вернуть {condition_id: [rules]} из списка чанков одного логического КП."""
    by_condition: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        text = chunk.get("text") or ""
        if len(text) < 40:
            continue

        for cid, pat in CONDITION_DIAG_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            tail = text[m.start() : m.start() + 2500]
            inc = RE_INCLUDES_BULLETS.search(tail)
            block = inc.group(1) if inc else tail[m.end() - m.start() : m.end() - m.start() + 800]
            components = _parse_required_components(block)
            if not components:
                continue
            rule = {
                "rule_id": f"auto_{cid}_diagnosis_formula",
                "rule_type": "diagnosis_formula",
                "required_components": components,
                "severity": "warning",
                "description_ru": f"Автоизвлечение: полнота формулировки диагноза ({cid}).",
                "source": _rule_source(chunk, protocol_id),
                "auto_extracted": True,
            }
            by_condition.setdefault(cid, []).append(rule)

        for cid, pat in CONDITION_CRIT_PATTERNS:
            if not pat.search(text):
                continue
            criteria: list[dict[str, str]] = []
            if cid == "gerd":
                sm = RE_GERD_SYMPTOM_CRIT.search(text)
                if sm:
                    criteria.append(
                        {
                            "symptom": "изжога или кислая регургитация",
                            "duration": ">= 6 месяцев",
                            "frequency": ">= 2 раза в неделю",
                        }
                    )
                low = text.lower()
                if "эндоскоп" in low or "рефлюкс-эзофагит" in low:
                    criteria.append(
                        {"finding": "рефлюкс-эзофагит", "method": "эндоскопия"}
                    )
                if "рн-метр" in low or "импеданс" in low:
                    criteria.append({"finding": "рН-метрия", "method": "рН-метрия"})
            if not criteria:
                continue
            rule = {
                "rule_id": f"auto_{cid}_diagnostic_criteria",
                "rule_type": "diagnostic_criterion",
                "logic": "any_of",
                "criteria": criteria,
                "severity": "warning",
                "description_ru": f"Автоизвлечение: диагностические критерии ({cid}).",
                "source": _rule_source(chunk, protocol_id),
                "auto_extracted": True,
            }
            by_condition.setdefault(cid, []).append(rule)

    return by_condition


def load_chunks_for_source(
    chunks_path: Path,
    source_substr: str,
    *,
    logical_suffix: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not chunks_path.is_file():
        return out
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = c.get("source_path") or ""
            if source_substr not in sp:
                continue
            if logical_suffix and not str(c.get("doc_id") or "").endswith(logical_suffix):
                continue
            out.append(c)
    return out


def merge_rules_into_gastro_mvp(
    extracted: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, int]:
    """Записать/обновить data/gastro_mvp/rules/auto_<condition>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for cid, rules in extracted.items():
        path = out_dir / f"auto_{cid}.json"
        manual_path = out_dir / f"{cid}_rules.json"
        manual_rules: list[dict] = []
        if manual_path.is_file():
            try:
                manual_rules = list(json.loads(manual_path.read_text(encoding="utf-8")).get("rules") or [])
            except Exception:
                manual_rules = []
        seen_ids = {r.get("rule_id") for r in manual_rules}
        merged = list(manual_rules)
        for r in rules:
            if r.get("rule_id") not in seen_ids:
                merged.append(r)
                seen_ids.add(r.get("rule_id"))
        payload = {"condition_id": cid, "rules": merged, "auto_merged": True}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[cid] = len(rules)
    return counts
