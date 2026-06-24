"""Сборка review для L2-fast без LLM synthesize."""
from __future__ import annotations

import re
from typing import Any


def template_summary_ru(structured_analysis: dict[str, Any] | None) -> str:
    if not isinstance(structured_analysis, dict):
        return "Структурный разбор завершён. Детали — в карточках согласования и выдержках протокола."
    comp = structured_analysis.get("compliance") or {}
    issues = comp.get("critical_issues") or comp.get("issues") or []
    lines: list[str] = []
    for item in issues[:3]:
        if isinstance(item, dict):
            txt = str(
                item.get("message_ru") or item.get("text") or item.get("issue") or ""
            ).strip()
        else:
            txt = str(item).strip()
        if txt:
            lines.append(f"• {txt}")
    if lines:
        return "Ключевые замечания:\n" + "\n".join(lines)
    score = comp.get("overall_score")
    status = comp.get("overall_status")
    if isinstance(score, (int, float)):
        st = f" ({status})" if status else ""
        return f"Структурная оценка {int(round(score))}%{st}. Сверка с протоколом — в карточках и evidence pack."
    return "Структурный разбор и сверка с протоколом выполнены без языковой модели."


def extract_block_gaps(alignment_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(alignment_result, dict):
        return []
    from clinical_knowledge.consult_evidence_quality import (
        is_kp_checklist_item,
        normalize_gap_text,
    )

    gaps: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    def _add(block_id: str, name_ru: str | None, gap_ru: str, *, low_score: bool = False) -> None:
        txt = normalize_gap_text(gap_ru)
        if not txt:
            return
        key = re.sub(r"\s+", " ", txt.lower())[:120]
        if key in seen_text:
            return
        seen_text.add(key)
        gaps.append(
            {
                "block_id": block_id,
                "gap_ru": txt,
                "name_ru": name_ru,
                "low_score": low_score,
            }
        )

    for card in alignment_result.get("alignment_cards") or []:
        if not isinstance(card, dict):
            continue
        block_id = str(card.get("block_id") or "")
        name_ru = card.get("name_ru")
        comment = str(card.get("comment_ru") or "").strip()
        score = card.get("score_pct")

        bullet_gaps: list[str] = []
        for g in card.get("gaps_ru") or []:
            txt = normalize_gap_text(str(g))
            if txt and is_kp_checklist_item(txt):
                bullet_gaps.append(txt)

        if block_id in ("exams", "treatment"):
            if comment:
                _add(block_id, name_ru, comment)
            for txt in bullet_gaps[:4]:
                if comment and txt.lower() in comment.lower():
                    continue
                _add(block_id, name_ru, txt)
            continue

        for txt in bullet_gaps[:4]:
            _add(block_id, name_ru, txt)

        if isinstance(score, (int, float)) and score < 50 and comment:
            if not any(normalize_gap_text(comment) in normalize_gap_text(x.get("gap_ru", "")) for x in gaps if x.get("block_id") == block_id):
                _add(block_id, name_ru, comment, low_score=True)

    return gaps[:10]


def protocol_paths_used(
    match_paths: list[str],
    alignment_result: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    align_paths: list[str] = []
    if isinstance(alignment_result, dict):
        audit = alignment_result.get("audit_trail") or {}
        if isinstance(audit, dict):
            align_paths = [str(p) for p in (audit.get("protocol_paths") or []) if p]
    for ps, reason in (
        *((str(p).strip(), "l1_match") for p in match_paths if p),
        *((str(p).strip(), "alignment") for p in align_paths if p),
    ):
        if not ps or ps in seen:
            continue
        seen.add(ps)
        out.append({"path": ps, "reason": reason})
    if isinstance(evidence_pack, dict):
        blocks = evidence_pack.get("blocks") or {}
        if isinstance(blocks, dict):
            for block_items in blocks.values():
                if not isinstance(block_items, list):
                    continue
                for it in block_items:
                    if not isinstance(it, dict):
                        continue
                    ps = str(it.get("protocol_path") or "").strip()
                    if ps and ps not in seen:
                        seen.add(ps)
                        out.append({"path": ps, "reason": "evidence"})
    return out


def build_l2_fast_review(
    *,
    structured_analysis: dict[str, Any] | None,
    alignment_result: dict[str, Any] | None,
) -> dict[str, Any]:
    review: dict[str, Any] = {
        "summary_ru": template_summary_ru(structured_analysis),
        "criteria": [],
        "limitations_ru": "",
        "disclaimer_ru": "Оценка ориентировочная; не замена МЭЭ и очной экспертизы.",
        "criteria_source": "deterministic_l2_fast",
    }
    if isinstance(alignment_result, dict) and (alignment_result.get("limitations_ru") or "").strip():
        review["limitations_ru"] = alignment_result["limitations_ru"]
    else:
        review["limitations_ru"] = (
            "L2: сверка с протоколом по детерминированным правилам, без оценки языковой модели."
        )
    return review
