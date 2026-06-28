"""LLM-бэкенды для offline chunk QA (Gemini / Ollama)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def backend_name() -> str:
    return (os.environ.get("CHUNK_QA_LLM_BACKEND") or "auto").strip().lower()


def _ollama_base() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return (os.environ.get("OLLAMA_MODEL") or "qwen2.5:3b").strip()


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{_ollama_base()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def generate_llm_text(prompt: str, *, max_out: int = 4000) -> str:
    """Вызов LLM по CHUNK_QA_LLM_BACKEND: auto | gemini | ollama."""
    mode = backend_name()
    if mode == "ollama" or (mode == "auto" and ollama_available()):
        return _generate_ollama(prompt, max_out=max_out)
    return _generate_gemini(prompt, max_out=max_out)


def _generate_ollama(prompt: str, *, max_out: int) -> str:
    model = _ollama_model()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_out},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base()}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("CHUNK_QA_OLLAMA_TIMEOUT", "180"))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err[:500]}") from e
    return str(body.get("response") or "")


def _generate_gemini(prompt: str, *, max_out: int) -> str:
    from rag_server import _extract_gemini_text, generate_gemini_consult_review_synthesize, get_gemini

    model = get_gemini()
    if model is None:
        raise RuntimeError("Gemini model unavailable (no API key)")
    resp = generate_gemini_consult_review_synthesize(model, prompt, max_out=max_out)
    return _extract_gemini_text(resp)
