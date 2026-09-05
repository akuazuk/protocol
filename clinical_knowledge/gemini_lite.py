"""Лёгкий клиент Gemini без импорта rag_server (offline batch / LLM extract)."""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any


def _extract_text(resp: Any) -> str:
    try:
        t = resp.text
        if t:
            return str(t).strip()
    except (ValueError, AttributeError, TypeError):
        pass
    parts: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts).strip()


def gemini_available() -> bool:
    from clinical_knowledge.gemini_client import available

    return available()


def get_lite_gemini_model():
    from clinical_knowledge.gemini_client import build_model
    from clinical_knowledge.gemini_model_config import main_gemini_model_name

    name, _warn = main_gemini_model_name()
    return build_model(name)


def generate_lite_json_response(model, full_prompt: str, *, timeout: float | None = None):
    """JSON-mode response with usage metadata and a hard timeout."""
    if timeout is None:
        timeout = float(os.environ.get("GEMINI_CALL_TIMEOUT", "240"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    max_out = int(os.environ.get("GEMINI_SUMMARY_EXTRACT_MAX_TOKENS", "8192"))

    def _run():
        return model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_out,
                response_mime_type="application/json",
            ),
        )

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            resp = fut.result(timeout=timeout)
        except FuturesTimeout as e:
            raise TimeoutError(f"Gemini timeout {timeout}s") from e
    return resp


def generate_lite_json(model, full_prompt: str, *, timeout: float | None = None) -> str:
    """JSON text compatibility wrapper for callers that do not need usage."""
    return _extract_text(generate_lite_json_response(model, full_prompt, timeout=timeout))
