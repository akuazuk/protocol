"""
Проверка, что GOOGLE_API_KEY из окружения работает и модель отвечает на запрос.
Связано с GET /api/verify-key в rag_server.
"""
from __future__ import annotations

import os


def _extract_text(resp) -> str:
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


def _diagnose_empty_response(resp) -> str:
    """Почему нет текста: блокировка, finish_reason и т.д."""
    bits: list[str] = []
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None:
        br = getattr(pf, "block_reason", None)
        if br is not None:
            bits.append(f"prompt_block_reason={br}")
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        bits.append("candidates=0")
    for i, c in enumerate(cands[:2]):
        fr = getattr(c, "finish_reason", None)
        if fr is not None:
            bits.append(f"candidate[{i}].finish_reason={fr}")
        idx = getattr(c, "index", None)
        if idx is not None:
            bits.append(f"candidate_index={idx}")
    return "; ".join(bits) if bits else "нет деталей от API"


def verify_gemini_key() -> tuple[bool, str]:
    """
    Возвращает (успех, сообщение).
    При успехе сообщение - короткий ответ модели (превью).
    """
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "Не задан ключ API на сервере (см. .env.example)."

    try:
        from clinical_knowledge.gemini_client import build_model
    except ImportError:
        return (
            False,
            "Не установлены зависимости сервера для обработки текста. "
            "Выполните: pip install -r requirements-rag.txt "
            "(тот же интерпретатор Python, что запускает uvicorn).",
        )

    try:
        name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        # Нейтральный промпт - фразы про «медицину» иногда дают пустой ответ из‑за фильтров
        model = build_model(name)
        r = model.generate_content(
            "Ответь одним словом: да.",
            generation_config={"max_output_tokens": 32, "temperature": 0},
        )
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return (
                False,
                "Лимит запросов к API (429 / quota). Подождите минуту или проверьте квоту и биллинг у поставщика API.",
            )
        return False, f"Ошибка API: {e!s}"

    text = _extract_text(r)
    if not text:
        detail = _diagnose_empty_response(r)
        return (
            False,
            "Пустой ответ модели. "
            + detail
            + ". Проверьте имя модели в конфигурации сервера (.env).",
        )

    return True, text
