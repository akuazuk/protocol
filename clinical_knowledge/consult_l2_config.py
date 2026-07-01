"""Конфигурация L2 lite: fast / evidence / narrative."""
from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _render_lite_enabled() -> bool:
    raw = os.environ.get("CONSULT_RENDER_L2_LITE")
    if raw is not None and str(raw).strip():
        return _env_bool("CONSULT_RENDER_L2_LITE", True)
    if _env_bool("CONSULT_REVIEW_FAST", False):
        return True
    profile = (os.environ.get("CONSULT_REVIEW_PROFILE") or "").strip().lower()
    if profile == "fast":
        return True
    if profile == "full":
        return False
    return _env_bool("RENDER", False)


def consult_l2_fast_enabled() -> bool:
    """Без synthesize Gemini: детерминированный L2 (по умолчанию при lite на Render)."""
    raw = os.environ.get("CONSULT_L2_FAST")
    if raw is not None and str(raw).strip():
        return _env_bool("CONSULT_L2_FAST", True)
    return _render_lite_enabled()


def consult_l2_narrative_requested(*, request_flag: bool = False) -> bool:
    if request_flag:
        return True
    return _env_bool("CONSULT_L2_NARRATIVE", False)


def consult_l2_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def consult_l2_align_max_paths() -> int:
    return consult_l2_env_int("CONSULT_L2_ALIGN_MAX_PATHS", 3)


def consult_l2_align_max_chunks_per_path() -> int:
    return consult_l2_env_int("CONSULT_L2_ALIGN_MAX_CHUNKS_PER_PATH", 4)


def consult_l2_evidence_max_paths() -> int:
    return consult_l2_env_int("CONSULT_L2_EVIDENCE_MAX_PATHS", 3)


def consult_l2_evidence_chunks_per_path() -> int:
    return consult_l2_env_int("CONSULT_L2_EVIDENCE_MAX_CHUNKS_PER_PATH", 2)


def consult_l2_evidence_max_chars() -> int:
    return consult_l2_env_int("CONSULT_L2_EVIDENCE_MAX_CHARS", 8000)


def consult_l2_evidence_summary_only() -> bool:
    """Выдержки L2 только из структурированных сводок (без сырых чанков PDF).

    По умолчанию включено: где нет валидной сводки по блоку - блок пуст,
    вместо мусорных фрагментов (колонтитулы, преамбулы, OCR, оргтекст).
    """
    return _env_bool("CONSULT_L2_EVIDENCE_SUMMARY_ONLY", True)


def resolve_l2_mode(*, narrative: bool = False) -> str:
    if narrative:
        return "narrative"
    return "fast" if consult_l2_fast_enabled() else "evidence"
