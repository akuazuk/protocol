"""Извлечение правил из размеченного корпуса (chunks.jsonl) — эвристики по тексту КП."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# (condition_id, regex для поиска блока «формулировка диагноза …»)
CONDITION_DIAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gerd", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+гэрб", re.I)),
    ("peptic_ulcer", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+гастродуоденальн", re.I)),
    ("gastritis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+хроническ", re.I)),
    ("functional_dyspepsia", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+функциональн", re.I)),
    ("ulcerative_colitis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+як\b", re.I)),
    ("ulcerative_colitis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+.*язвенн.*колит", re.I)),
    ("crohn", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+.*(?:крон|болезн[иь]\s+крона)", re.I)),
    ("celiac", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+.*целиак", re.I)),
    ("celiac", re.compile(r"при\s+формулировк[аи]\s+диагноз[а]?\s+.*целиак", re.I)),
]

CONDITION_CRIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gerd", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+гэрб", re.I)),
    ("peptic_ulcer", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+гастродуоденальн", re.I)),
    ("gastritis", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+хроническ", re.I)),
    ("ulcerative_colitis", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+як\b", re.I)),
    ("crohn", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+.*крон", re.I)),
    ("celiac", re.compile(r"диагностическ(?:им|ими)\s+критери(?:ем|ями)\s+.*целиак", re.I)),
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
        "source_path": chunk.get("source_path"),
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
    rule_id_prefix: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Вернуть {condition_id: [rules]} из списка чанков одного логического КП."""
    prefix = (rule_id_prefix + "_") if rule_id_prefix else ""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    seen_rule_keys: set[str] = set()

    def _add_rule(cid: str, rule: dict[str, Any]) -> None:
        key = f"{cid}:{rule.get('rule_type')}:{rule.get('rule_id')}"
        if key in seen_rule_keys:
            return
        seen_rule_keys.add(key)
        by_condition.setdefault(cid, []).append(rule)

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
                "rule_id": f"{prefix}auto_{cid}_diagnosis_formula",
                "rule_type": "diagnosis_formula",
                "required_components": components,
                "severity": "warning",
                "description_ru": f"Автоизвлечение: полнота формулировки диагноза ({cid}).",
                "source": _rule_source(chunk, protocol_id),
                "auto_extracted": True,
            }
            _add_rule(cid, rule)

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
                "rule_id": f"{prefix}auto_{cid}_diagnostic_criteria",
                "rule_type": "diagnostic_criterion",
                "logic": "any_of",
                "criteria": criteria,
                "severity": "warning",
                "description_ru": f"Автоизвлечение: диагностические критерии ({cid}).",
                "source": _rule_source(chunk, protocol_id),
                "auto_extracted": True,
            }
            _add_rule(cid, rule)

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


def load_chunks_exact(chunks_path: Path, source_path: str) -> list[dict[str, Any]]:
    """Все чанки одного PDF (точное совпадение source_path)."""
    want = source_path.replace("\\", "/")
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
            sp = (c.get("source_path") or "").replace("\\", "/")
            if sp == want:
                out.append(c)
    return out


def _score_logical_doc(chunks: list[dict[str, Any]]) -> int:
    blob = " ".join((c.get("text") or "") for c in chunks)
    score = 0
    for _, pat in CONDITION_DIAG_PATTERNS:
        if pat.search(blob):
            score += 3
    for _, pat in CONDITION_CRIT_PATTERNS:
        if pat.search(blob):
            score += 2
    if "формулировк" in blob.lower():
        score += 1
    return score


def pick_best_logical_chunks(all_pdf_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Выбрать логический документ внутри PDF с наибольшей клинической разметкой."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in all_pdf_chunks:
        by_doc[str(c.get("doc_id") or "")].append(c)
    if not by_doc:
        return []
    ranked = sorted(
        by_doc.items(),
        key=lambda kv: (_score_logical_doc(kv[1]), len(kv[1])),
        reverse=True,
    )
    best_score = _score_logical_doc(ranked[0][1])
    if best_score <= 0:
        ranked = sorted(by_doc.items(), key=lambda kv: len(kv[1]), reverse=True)
    return ranked[0][1]


def gastro_source_paths(registry_jsonl: Path) -> list[str]:
    paths: set[str] = set()
    if not registry_jsonl.is_file():
        return []
    for line in registry_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("specialty_slug") != "gastroenterologiya":
            continue
        sp = (row.get("source_path") or "").replace("\\", "/")
        if sp:
            paths.add(sp)
    return sorted(paths)


def _unique_pdf_paths_from_registry(registry_jsonl: Path) -> list[str]:
    """Один путь на PDF (без дублей логических частей)."""
    seen_pdf: set[str] = set()
    out: list[str] = []
    for sp in gastro_source_paths(registry_jsonl):
        if sp in seen_pdf:
            continue
        seen_pdf.add(sp)
        out.append(sp)
    return out


def extract_rules_all_gastro_pdfs(
    chunks_path: Path,
    registry_jsonl: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Извлечь правила по всем уникальным гастро-PDF."""
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_pdf: dict[str, Any] = {}
    pdfs = _unique_pdf_paths_from_registry(registry_jsonl)
    for sp in pdfs:
        all_ch = load_chunks_exact(chunks_path, sp)
        if not all_ch:
            per_pdf[sp] = {"chunks": 0, "rules": 0, "skipped": "no_chunks"}
            continue
        doc_chunks = pick_best_logical_chunks(all_ch)
        pdf_hash = sha256(sp.encode()).hexdigest()[:8]
        protocol_id = f"gastro_{pdf_hash}"
        extracted = extract_rules_from_chunks(
            doc_chunks,
            protocol_id=protocol_id,
            rule_id_prefix=pdf_hash,
        )
        n_rules = sum(len(v) for v in extracted.values())
        per_pdf[sp] = {
            "chunks": len(doc_chunks),
            "rules": n_rules,
            "doc_id": doc_chunks[0].get("doc_id") if doc_chunks else None,
        }
        for cid, rules in extracted.items():
            merged[cid].extend(rules)
    return dict(merged), {"pdfs_total": len(pdfs), "pdfs": per_pdf}


def merge_rules_into_gastro_mvp(
    extracted: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, int]:
    """Записать data/gastro_mvp/rules/auto_<condition>.json (только авто-правила)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for cid, rules in extracted.items():
        path = out_dir / f"auto_{cid}.json"
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for r in rules:
            rid = str(r.get("rule_id") or "")
            if rid and rid in seen:
                continue
            if rid:
                seen.add(rid)
            deduped.append(r)
        payload = {"condition_id": cid, "rules": deduped, "auto_only": True}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[cid] = len(deduped)
    return counts
