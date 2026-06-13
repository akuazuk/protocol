"""Сопоставление КЗ с Protocol Summary Cards (без точного protocol_id)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .loader import (
    find_conditions_by_icd,
    find_conditions_by_text,
    find_summary_for_condition,
    load_protocol_summaries,
    load_summary_by_protocol_id,
)
from .schema import ConditionSummary, ProtocolSummary
from .validator import summary_is_usable

ROOT = Path(__file__).resolve().parents[2]


def _norm_path(p: str | None) -> str:
    if not p:
        return ""
    return p.replace("\\", "/").strip().lower()


def _path_match(a: str | None, b: str | None) -> bool:
    na, nb = _norm_path(a), _norm_path(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return na.endswith(nb) or nb.endswith(na) or Path(na).name == Path(nb).name


def _title_sim(a: str, b: str) -> float:
    a = re.sub(r"\s+", " ", (a or "").lower()).strip()
    b = re.sub(r"\s+", " ", (b or "").lower()).strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:120], b[:120]).ratio()


def _add_summary(
    out: dict[str, ProtocolSummary],
    diagnostics: list[dict[str, Any]],
    summary: ProtocolSummary,
    *,
    reason: str,
    score: float,
    condition_ids: list[str],
) -> None:
    if usable := summary:
        pid = usable.protocol_id
        if pid not in out:
            out[pid] = usable
            diagnostics.append({
                "protocol_id": pid,
                "title": usable.source.title,
                "match_reasons": [reason],
                "match_score": score,
                "condition_ids": list(condition_ids),
            })
        else:
            for d in diagnostics:
                if d.get("protocol_id") == pid:
                    if reason not in d["match_reasons"]:
                        d["match_reasons"].append(reason)
                    d["match_score"] = max(float(d.get("match_score") or 0), score)
                    for cid in condition_ids:
                        if cid not in d["condition_ids"]:
                            d["condition_ids"].append(cid)
                    break


def discover_protocol_summaries(
    *,
    icd_codes: list[str],
    diagnosis_texts: list[str],
    matched_protocols: list[dict[str, Any]] | None = None,
    specialty_slug: str | None = None,
    usable_only: bool = True,
) -> tuple[list[ProtocolSummary], list[dict[str, Any]], list[str]]:
    """Подбор summary по ICD, тексту диагноза, legacy match (path/sha/title/rubric).

    Returns: (summaries, diagnostics, matched_condition_ids)
    """
    matched_protocols = matched_protocols or []
    found: dict[str, ProtocolSummary] = {}
    diagnostics: list[dict[str, Any]] = []
    condition_ids: list[str] = []

    def _usable(s: ProtocolSummary | None) -> ProtocolSummary | None:
        if s is None:
            return None
        if usable_only and not summary_is_usable(s):
            return None
        return s

    # 1) Точный protocol_id из legacy match
    for m in matched_protocols:
        pid = str(m.get("protocol_id") or m.get("card_id") or "")
        if pid:
            s = _usable(load_summary_by_protocol_id(pid))
            if s:
                cids = [c.condition_id for c in s.conditions]
                _add_summary(found, diagnostics, s, reason="protocol_id", score=1.0, condition_ids=cids)
                condition_ids.extend(cids)

    # 2) local_path / sha256 / title / rubric из legacy cards
    for m in matched_protocols:
        m_path = m.get("source_path") or m.get("local_path") or ""
        m_sha = m.get("document_sha256") or m.get("sha256")
        m_title = str(m.get("document_title") or m.get("title") or "")
        m_rubric = str(m.get("category_slug") or m.get("specialty_slug") or m.get("rubric_slug") or "")

        for summary in load_protocol_summaries(usable_only=usable_only):
            if summary.protocol_id in found:
                continue
            # legacy match по path/sha/title/rubric
            reasons: list[str] = []
            score = 0.0
            if _path_match(summary.source.local_path, m_path):
                reasons.append("source_path")
                score = max(score, 0.95)
            if m_sha and summary.source.document_sha256 and m_sha == summary.source.document_sha256:
                reasons.append("sha256")
                score = max(score, 0.98)
            ts = _title_sim(summary.source.title, m_title)
            if ts >= 0.55:
                reasons.append(f"title_similarity:{ts:.2f}")
                score = max(score, ts)
            if specialty_slug and summary.rubric.slug and summary.rubric.slug == specialty_slug:
                reasons.append("rubric_slug")
                score = max(score, 0.7)
            elif m_rubric and summary.rubric.slug and summary.rubric.slug == m_rubric:
                reasons.append("legacy_rubric")
                score = max(score, 0.65)
            if reasons:
                cids = [c.condition_id for c in summary.conditions]
                _add_summary(
                    found, diagnostics, summary,
                    reason=",".join(reasons), score=score, condition_ids=cids,
                )
                condition_ids.extend(cids)

    # 3) ICD codes
    for code in icd_codes:
        code = str(code or "").strip()
        if not code:
            continue
        for cond in find_conditions_by_icd(code):
            condition_ids.append(cond.condition_id)
            summary = _usable(find_summary_for_condition(cond, usable_only=usable_only))
            if summary:
                _add_summary(
                    found, diagnostics, summary,
                    reason=f"icd10:{code}", score=0.85, condition_ids=[cond.condition_id],
                )

    # 4) Текст диагнозов
    for raw in diagnosis_texts:
        text = re.sub(r"\s+", " ", (raw or "").strip())
        if len(text) < 3:
            continue
        # убрать код МКБ из начала для текстового поиска
        q = re.sub(r"^[A-Z]\d{2}(?:\.\d+)?\s*", "", text, flags=re.I).strip() or text
        for cond in find_conditions_by_text(q, limit=8):
            condition_ids.append(cond.condition_id)
            summary = _usable(find_summary_for_condition(cond, usable_only=usable_only))
            if summary:
                _add_summary(
                    found, diagnostics, summary,
                    reason=f"diagnosis_text:{q[:40]}", score=0.75, condition_ids=[cond.condition_id],
                )

    # Фильтр по рубрике (мягкий): если slug задан и summary другой рубрики без ICD match - понизить
    if specialty_slug and found:
        filtered: dict[str, ProtocolSummary] = {}
        for pid, s in found.items():
            diag = next((d for d in diagnostics if d.get("protocol_id") == pid), {})
            reasons = diag.get("match_reasons") or []
            if any(r.startswith("icd10:") for r in reasons):
                filtered[pid] = s
                continue
            if s.rubric.slug and s.rubric.slug != specialty_slug:
                continue
            filtered[pid] = s
        if filtered:
            found = filtered

    summaries = list(found.values())
    condition_ids = list(dict.fromkeys(condition_ids))
    if not summaries and (icd_codes or diagnosis_texts):
        diagnostics.append({
            "protocol_id": None,
            "title": None,
            "match_reasons": ["not_found"],
            "match_score": 0.0,
            "condition_ids": [],
            "detail": (
                f"Summary не найден по ICD={icd_codes[:6]} "
                f"и диагнозам={diagnosis_texts[:2]}"
            ),
        })
    return summaries, diagnostics, condition_ids
