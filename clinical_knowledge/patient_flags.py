"""Feature flags для B2C patient report v2."""
from __future__ import annotations

import os


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def patient_report_v2_enabled() -> bool:
    return _flag("PATIENT_REPORT_V2_ENABLED", "1")


def patient_protocol_age_filter_enabled() -> bool:
    return _flag("PATIENT_PROTOCOL_AGE_FILTER_ENABLED", "1")


def patient_safe_quotes_enabled() -> bool:
    return _flag("PATIENT_SAFE_QUOTES_ENABLED", "1")


def patient_question_safety_enabled() -> bool:
    return _flag("PATIENT_QUESTION_SAFETY_ENABLED", "1")


def patient_plain_terms_enabled() -> bool:
    return _flag("PATIENT_PLAIN_TERMS_ENABLED", "1")


def patient_visit_sheet_pdf_enabled() -> bool:
    return _flag("PATIENT_VISIT_SHEET_PDF_ENABLED", "1")


def patient_no_history_mode_enabled() -> bool:
    return _flag("PATIENT_NO_HISTORY_MODE_ENABLED", "0")


def patient_show_protocol_technical_block() -> bool:
    return _flag("PATIENT_SHOW_PROTOCOL_TECHNICAL_BLOCK", "0")


def patient_onco_questions_enabled() -> bool:
    """B2C-блок «вопросы врачу» из онконастороженности (по умолчанию вкл, см. render.yaml)."""
    return _flag("PATIENT_ONCO_QUESTIONS_ENABLED", "1")


def patient_rag_retrieval_enabled() -> bool:
    """B2C: RAG + vector index для подбора протоколов (corpus_chunks_parts + embeddings)."""
    return _flag("PATIENT_RAG_RETRIEVAL_ENABLED", "1")


def patient_rag_semantic_fallback_enabled() -> bool:
    """B2C: повторный RAG без path_allowlist при слабом L1 (0 протоколов или низкий confidence)."""
    return _flag("PATIENT_RAG_SEMANTIC_FALLBACK", "1")


def patient_rag_questions_enabled() -> bool:
    """B2C: дополнительные вопросы врачу из RAG-чанков при weak L1 или <3 вопросов."""
    return _flag("PATIENT_RAG_QUESTIONS_ENABLED", "1")


def patient_upload_semantic_rescue_enabled() -> bool:
    """B2C: semantic probe вместо отказа классификатора (empty/unknown)."""
    return _flag("PATIENT_UPLOAD_SEMANTIC_RESCUE", "0")
