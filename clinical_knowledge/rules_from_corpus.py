"""Извлечение правил из размеченного корпуса (chunks.jsonl) - эвристики по тексту КП."""
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
    ("gerd", re.compile(r"формулировк[аи]\s+основного\s+диагноз", re.I)),
    ("peptic_ulcer", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+гастродуоденальн", re.I)),
    ("gastritis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+хроническ", re.I)),
    ("gastritis", re.compile(r"формулировк[аи]\s+диагноз[а]?\s+хг\b", re.I)),
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

# Подсказка нозологии из пути PDF (fallback, если нет «формулировка диагноза»).
SOURCE_PATH_CONDITION_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("ulcerative_colitis", ("язвенн", "колит", "k51")),
    ("crohn", ("крон", "k50", "энтерит")),
    ("celiac", ("целиак", "k90")),
    ("gerd", ("гэрб", "рефлюкс", "пищевода_желудка")),
    ("gastritis", ("гастрит", "k29")),
    ("peptic_ulcer", ("язвенн", "гастродуоден")),
    ("functional_dyspepsia", ("диспепс", "k30")),
    ("acute_pancreatitis", ("панкреат", "k85")),
    ("acute_appendicitis", ("аппендицит", "k35", "k37")),
]

# Нумерованные разделы «компонентов диагноза» в детских КП ЯК/БК.
IBD_NUMBERED_DIAG_SECTIONS: dict[str, tuple[str, ...]] = {
    "ulcerative_colitis": ("13.1", "13.2", "13.4"),
    "crohn": ("16.1", "16.2"),
}

IBD_SECTION_COMPONENT_LABELS: dict[str, str] = {
    "13.1": "протяженность",
    "13.2": "фаза",
    "13.3": "характер течения",
    "13.4": "тяжесть",
    "13.5": "ответ на терапию",
    "13.6": "осложнения",
    "13.7": "хирургическое лечение",
    "16.1": "локализация",
    "16.2": "фаза",
    "16.3": "характер течения",
    "16.4": "ответ на терапию",
    "16.5": "осложнения",
    "16.6": "хирургическое лечение",
}

RE_INCLUDES_BULLETS = re.compile(
    r"включает\s*:?\s*(.+?)(?=пример|^\d+\.|диагностическ|национальн|\Z)",
    re.I | re.S | re.M,
)
RE_GERD_SYMPTOM_CRIT = re.compile(
    r"изжога\s+или\s+кислая\s+регургитация[^.\n]{0,120}",
    re.I,
)
RE_NUMBERED_DIAG_ITEM = re.compile(
    r"(?:^|\s)(\d+\.\d+)\.\s*([^:\n]{4,120}):",
    re.M,
)
RE_GENERIC_DIAG_FORMULA = re.compile(
    r"формулировк[аи]\s+(?:основного\s+)?диагноз",
    re.I,
)


def _rubric_condition_id(source_path: str) -> str:
    parts = (source_path or "").replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0] == "minzdrav_protocols":
        return f"rubric_{parts[1].replace('-', '_')}"
    return "general_protocol"


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def infer_condition_ids_from_source_path(source_path: str) -> list[str]:
    low = (source_path or "").lower().replace("\\", "/")
    out: list[str] = []
    for cid, needles in SOURCE_PATH_CONDITION_HINTS:
        if any(n in low for n in needles):
            out.append(cid)
    try:
        from .condition_registry import CONDITIONS

        for c in CONDITIONS:
            if c.condition_id in out:
                continue
            if any(h in low for h in c.path_hints):
                out.append(c.condition_id)
        from .rules_from_path import infer_path_condition

        inferred = infer_path_condition(source_path)
        if inferred and inferred[0] not in out:
            out.append(inferred[0])
    except ImportError:
        pass
    return out


def _parse_numbered_diagnosis_components(blob: str, section_ids: tuple[str, ...]) -> list[str]:
    """Компоненты диагноза из пунктов 13.x / 16.x (детские КП ЯК/БК)."""
    norm = _collapse_ws(blob)
    components: list[str] = []
    for sec_id in section_ids:
        if re.search(rf"(?:^|\s){re.escape(sec_id)}\.", norm):
            label = IBD_SECTION_COMPONENT_LABELS.get(sec_id)
            if label and label not in components:
                components.append(label)
    if not components:
        return []
    if "нозология" not in components:
        components.insert(0, "нозология")
    return components[:10]


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
    source_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Вернуть {condition_id: [rules]} из списка чанков одного логического КП."""
    prefix = (rule_id_prefix + "_") if rule_id_prefix else ""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    seen_rule_keys: set[str] = set()
    blob = _collapse_ws(" ".join((c.get("text") or "") for c in chunks))
    anchor_chunk = chunks[0] if chunks else {}

    def _add_rule(cid: str, rule: dict[str, Any]) -> None:
        key = f"{cid}:{rule.get('rule_type')}:{rule.get('rule_id')}"
        if key in seen_rule_keys:
            return
        seen_rule_keys.add(key)
        by_condition.setdefault(cid, []).append(rule)

    for chunk in chunks:
        text = chunk.get("text") or ""
        text_norm = _collapse_ws(text)
        if len(text_norm) < 40:
            continue

        for cid, pat in CONDITION_DIAG_PATTERNS:
            m = pat.search(text_norm)
            if not m:
                continue
            tail = text_norm[m.start() : m.start() + 2500]
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
            if not pat.search(text_norm):
                continue
            criteria: list[dict[str, str]] = []
            if cid == "gerd":
                sm = RE_GERD_SYMPTOM_CRIT.search(text_norm)
                if sm:
                    criteria.append(
                        {
                            "symptom": "изжога или кислая регургитация",
                            "duration": ">= 6 месяцев",
                            "frequency": ">= 2 раза в неделю",
                        }
                    )
                low = text_norm.lower()
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

    path_cids = infer_condition_ids_from_source_path(source_path or "")
    for cid in path_cids:
        sections = IBD_NUMBERED_DIAG_SECTIONS.get(cid)
        if not sections:
            continue
        if any(r.get("rule_type") == "diagnosis_formula" for r in by_condition.get(cid, [])):
            continue
        components = _parse_numbered_diagnosis_components(blob, sections)
        if not components:
            continue
        src = _rule_source(anchor_chunk, protocol_id)
        if source_path:
            src["source_path"] = source_path.replace("\\", "/")
        rule = {
            "rule_id": f"{prefix}auto_{cid}_numbered_diagnosis_formula",
            "rule_type": "diagnosis_formula",
            "required_components": components,
            "severity": "warning",
            "description_ru": f"Автоизвлечение: компоненты диагноза по разделам КП ({cid}).",
            "source": src,
            "auto_extracted": True,
            "extraction_method": "numbered_sections",
        }
        _add_rule(cid, rule)

    if source_path and not by_condition:
        from .rules_from_path import infer_path_condition

        inferred = infer_path_condition(source_path)
        cid = inferred[0] if inferred else _rubric_condition_id(source_path)
        default_components = list(inferred[1]) if inferred else []
        for chunk in chunks:
            text_norm = _collapse_ws(chunk.get("text") or "")
            if len(text_norm) < 40 or not RE_GENERIC_DIAG_FORMULA.search(text_norm):
                continue
            m = RE_GENERIC_DIAG_FORMULA.search(text_norm)
            if not m:
                continue
            tail = text_norm[m.start() : m.start() + 2500]
            inc = RE_INCLUDES_BULLETS.search(tail)
            block = inc.group(1) if inc else tail[m.end() - m.start() : m.end() - m.start() + 800]
            components = _parse_required_components(block) or default_components
            if not components:
                components = ["нозология", "клиническая форма", "степень тяжести", "осложнения"]
            src = _rule_source(chunk, protocol_id)
            src["source_path"] = source_path.replace("\\", "/")
            rule = {
                "rule_id": f"{prefix}auto_{cid}_generic_diagnosis_formula",
                "rule_type": "diagnosis_formula",
                "required_components": components[:10],
                "severity": "warning",
                "description_ru": f"Автоизвлечение: формулировка диагноза ({cid}).",
                "source": src,
                "auto_extracted": True,
                "extraction_method": "corpus_generic",
            }
            _add_rule(cid, rule)
            break

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
    blob = _collapse_ws(" ".join((c.get("text") or "") for c in chunks))
    score = 0
    rich_types = frozenset({
        "diagnostics", "criteria_block", "table", "pharmacotherapy",
        "treatment", "drug_list", "dispensary", "prevention",
    })
    for c in chunks:
        ct = (c.get("chunk_type") or c.get("kind") or "").strip().lower()
        if ct in rich_types:
            score += 1
    for _, pat in CONDITION_DIAG_PATTERNS:
        if pat.search(blob):
            score += 3
    for _, pat in CONDITION_CRIT_PATTERNS:
        if pat.search(blob):
            score += 2
    if "формулировк" in blob.lower():
        score += 1
    if RE_NUMBERED_DIAG_ITEM.search(blob):
        score += 2
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
            source_path=sp,
        )
        n_rules = sum(len(v) for v in extracted.values())
        extraction_method = None
        if n_rules == 0:
            from .rules_from_path import extract_path_rules

            path_rules = extract_path_rules(
                sp, protocol_id=protocol_id, rule_id_prefix=pdf_hash
            )
            for cid, rules in path_rules.items():
                extracted.setdefault(cid, []).extend(rules)
            n_rules = sum(len(v) for v in extracted.values())
            if n_rules:
                extraction_method = "path_template"
        per_pdf[sp] = {
            "chunks": len(doc_chunks),
            "rules": n_rules,
            "doc_id": doc_chunks[0].get("doc_id") if doc_chunks else None,
            "extraction_method": extraction_method,
        }
        for cid, rules in extracted.items():
            merged[cid].extend(rules)
    return dict(merged), {"pdfs_total": len(pdfs), "pdfs": per_pdf}


def merge_rules_into_gastro_mvp(
    extracted: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, int]:
    """Записать data/gastro_mvp/rules/auto_<condition>.json и path_<condition>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    auto_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cid, rules in extracted.items():
        for r in rules:
            if r.get("extraction_method") == "path_template":
                path_by_cid[cid].append(r)
            else:
                auto_by_cid[cid].append(r)

    def _write_bucket(prefix: str, bucket: dict[str, list[dict[str, Any]]]) -> None:
        for cid, rules in bucket.items():
            path = out_dir / f"{prefix}_{cid}.json"
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for r in rules:
                rid = str(r.get("rule_id") or "")
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                deduped.append(r)
            if not deduped:
                continue
            payload = {
                "condition_id": cid,
                "rules": deduped,
                f"{prefix}_only": True,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            counts[cid] = counts.get(cid, 0) + len(deduped)

    _write_bucket("auto", auto_by_cid)
    _write_bucket("path", path_by_cid)
    return counts
