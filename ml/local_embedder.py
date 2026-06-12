"""Локальный bi-encoder (e5) вместо cloud embed API для retrieve()."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, device="cpu")


def make_embed_fn(model_path: str) -> Callable[[str, str, str | None], list[float]]:
    """Совместим с сигнатурой rag_server._gemini_embed_one(model, text, task_type)."""

    def _embed_one(_model: str, text: str, task_type: str | None) -> list[float]:
        st = _load_model(model_path)
        t = (text or "").strip()[:8000]
        if task_type == "retrieval_query":
            prefixed = f"query: {t}"
        else:
            prefixed = f"passage: {t}"
        vec = st.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return [float(x) for x in vec]

    return _embed_one
