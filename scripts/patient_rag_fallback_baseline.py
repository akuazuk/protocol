#!/usr/bin/env python3
"""Снимок метрик run_patient_review для сравнения до/после RAG fallback."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge.patient_review import run_patient_review


def _ensure_rag_corpus() -> None:
    """Синхронная загрузка корпуса (вне uvicorn фоновый load не стартует)."""
    try:
        import rag_server as rs

        if not getattr(rs, "_chunks", None):
            rs.load_data()
        rs._chunks_load_done.set()
    except Exception:
        pass

FIXTURES = {
    "orvi_no_icd": (
        "Жалобы: кашель, насморк, температура 37.8.\n"
        "Диагноз: ОРВИ.\n"
        "Рекомендации: симптоматическое лечение, парацетамол при температуре, "
        "обильное питьё. Контроль через 7 дней."
    ),
    "short_kz": (
        "Диагноз: ОРВИ. Р: симптоматически, парацетамол. Контроль 7 дн."
    ),
    "noisy_ocr": (
        "Жaлoбы: кашeль 3 дня\n"
        "Диaгноз ОРВИ\n"
        "Рек0мендации: питьё, парацетамол\n"
        "Контроль"
    ),
    "good_kz_icd": (
        "Врач: флеболог\n"
        "Жалобы: отёк правой голени.\n"
        "Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.\n"
        "Рекомендации: ривароксабан 20 мг 1 раз в день.\n"
        "Контроль через 3 месяца."
    ),
}


def snapshot() -> dict:
    out: dict = {}
    for name, text in FIXTURES.items():
        try:
            r = run_patient_review(text=text, consultation_id=f"baseline-{name}")
        except Exception as e:
            out[name] = {"error": str(e)[:200]}
            continue
        pr = r.get("patient_report") or {}
        out[name] = {
            "upload_mismatch": bool(r.get("upload_mismatch")),
            "matched_protocols_count": r.get("matched_protocols_count"),
            "confidence_score": r.get("confidence_score"),
            "rag_used": r.get("rag_used"),
            "questions_count": len(pr.get("questions_structured") or pr.get("questions_for_doctor") or []),
            "citations_count": len(pr.get("protocol_citations") or []),
            "protocol_rag_meta": pr.get("protocol_rag_meta"),
            "document_quality": pr.get("document_quality"),
            "guessed_kind": r.get("guessed_kind"),
        }
    return out


if __name__ == "__main__":
    _ensure_rag_corpus()
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
