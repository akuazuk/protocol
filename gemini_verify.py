"""
Проверка, что GOOGLE_API_KEY из окружения работает и модель отвечает на запрос.
Связано с GET /api/verify-key в rag_server.
"""
from __future__ import annotations

import os
import warnings


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
    При успехе сообщение — короткий ответ модели (превью).
    """
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "Нет GOOGLE_API_KEY (добавьте в .env или export)."

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai
            from google.generativeai.types import HarmBlockThreshold, HarmCategory
    except ImportError:
        return (
            False,
            "Нет пакета google-generativeai в ЭТОМ интерпретаторе Python. "
            "Установите: python3 -m pip install google-generativeai "
            "или используйте тот же python, куда ставили зависимости (часто python3.11 на Mac).",
        )

    try:
        genai.configure(api_key=key)
        name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        # Нейтральный промпт — фразы про «медицину» иногда дают пустой ответ из‑за фильтров
        safety = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        ]
        model = genai.GenerativeModel(name, safety_settings=safety)
        r = model.generate_content(
            "Ответь одним словом: да.",
            generation_config={"max_output_tokens": 32, "temperature": 0},
        )
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return (
                False,
                "Лимит запросов к API (429 / quota). Подождите минуту или проверьте квоту и биллинг в Google AI Studio.",
            )
        return False, f"Ошибка API: {e!s}"

    text = _extract_text(r)
    if not text:
        detail = _diagnose_empty_response(r)
        return (
            False,
            "Пустой ответ модели. "
            + detail
            + ". Проверьте GEMINI_MODEL (нужен gemini-2.5-flash или новее для новых ключей).",
        )

    return True, text
