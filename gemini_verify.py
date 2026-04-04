"""
Проверка, что GOOGLE_API_KEY из окружения работает и модель отвечает на запрос.
Используется: check_gemini_key.py и GET /api/verify-key в rag_server.
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


def verify_gemini_key() -> tuple[bool, str]:
    """
    Возвращает (успех, сообщение).
    При успехе сообщение — короткий ответ модели (превью).
    """
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "Нет GOOGLE_API_KEY (добавьте в .env или export)."

    try:
        import google.generativeai as genai
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
        model = genai.GenerativeModel(name)
        r = model.generate_content(
            "Ответь одним словом «да», если получил запрос и готов отвечать на вопросы по медицинским текстам.",
            generation_config={"max_output_tokens": 40, "temperature": 0},
        )
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return (
                False,
                "Лимит запросов Gemini (429 / quota). Подождите минуту или проверьте квоту и биллинг в Google AI Studio.",
            )
        return False, f"Ошибка API: {e!s}"

    text = _extract_text(r)
    if not text:
        return False, "Пустой ответ модели (возможна блокировка или сбой)."

    return True, text
