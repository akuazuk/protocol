"""Лёгкий клиент Gemini без импорта rag_server (offline batch / LLM extract)."""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def thinking_budget() -> int:
    """0 = без thinking-токенов (они биллятся как output у Gemini 3.x)."""
    return _env_int("GEMINI_THINKING_BUDGET", 0)


def json_max_output_tokens() -> int:
    """Ночной JSON короче extract: runner задаёт GEMINI_JSON_MAX_OUTPUT_TOKENS=2048."""
    if (os.environ.get("GEMINI_JSON_MAX_OUTPUT_TOKENS") or "").strip():
        return _env_int("GEMINI_JSON_MAX_OUTPUT_TOKENS", 2048)
    return _env_int("GEMINI_SUMMARY_EXTRACT_MAX_TOKENS", 8192)


def is_terminal_billing_error(message: str) -> bool:
    """Spend cap / monthly billing - не ретраить: после смены ключа это платный объём."""
    low = (message or "").lower()
    return (
        "spending cap" in low
        or "monthly spend" in low
        or "exceeded its monthly" in low
        or "exceeded your current quota" in low
    )


def json_generation_config_dict(
    *,
    max_output_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Конфиг для ночного JSON: без thinking, короткий output.

    Словарь, а не GenerationConfig: так thinking_config доезжает даже на SDK,
    который ещё не знает поле, и Batch REST принимает тот же payload.
    """
    cfg: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": int(
            max_output_tokens if max_output_tokens is not None else json_max_output_tokens()
        ),
        "candidate_count": 1,
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_budget": thinking_budget()},
    }
    return cfg


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


def _generation_config_obj(genai, cfg: dict[str, Any]):
    try:
        return genai.types.GenerationConfig(**cfg)
    except TypeError:
        stripped = {k: v for k, v in cfg.items() if k != "thinking_config"}
        try:
            return genai.types.GenerationConfig(**stripped)
        except TypeError:
            return cfg


def generate_lite_json_response(
    model,
    full_prompt: str,
    *,
    timeout: float | None = None,
    max_output_tokens: int | None = None,
):
    """JSON-mode response with usage metadata and a hard timeout."""
    if timeout is None:
        timeout = float(os.environ.get("GEMINI_CALL_TIMEOUT", "240"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    cfg = json_generation_config_dict(max_output_tokens=max_output_tokens)
    generation_config = _generation_config_obj(genai, cfg)

    def _run():
        try:
            return model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
        except TypeError:
            # SDK отклонил thinking_config в объекте - второй заход словарём без него.
            return model.generate_content(
                full_prompt,
                generation_config=_generation_config_obj(
                    genai, {k: v for k, v in cfg.items() if k != "thinking_config"}
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
