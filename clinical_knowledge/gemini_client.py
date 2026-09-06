#!/usr/bin/env python3
"""Единая точка создания клиента Gemini.

Зачем модуль появился (2026-09-05): клиент собирался в шести местах, и
`safety_settings` выставляли только три из них. В частности
`scripts/grade_kz_llm.py` - оценка качества консультативных заключений -
создавал модель без них:

    return genai.GenerativeModel(name), name

Для клинического текста это не мелочь. По умолчанию Gemini блокирует по порогу
BLOCK_MEDIUM_AND_ABOVE, а в заключениях штатно встречаются травмы, самоповреждения,
насилие, сексуальное и репродуктивное здоровье, онкология. Ответ на такой текст
блокируется, генерация возвращает пустоту - и оценка либо теряется, либо
записывается битой, без ошибки в логе. Это тот же класс отказа, который в проекте
уже описан для geo-ошибок: «файл выглядит полным, но оценки битые».

Поэтому порог здесь один для всех клинических путей - BLOCK_ONLY_HIGH, то есть
режем только заведомо опасное, а клиническую лексику пропускаем. Настройка
собрана в одном месте, чтобы новый вызов не мог случайно уехать на дефолт.

Ключ читается из GOOGLE_API_KEY, затем GEMINI_API_KEY - порядок такой же, каким
он сложился во всех прежних местах, чтобы поведение окружений не поменялось.
"""

from __future__ import annotations

import os
import warnings
from typing import Any


class GeminiKeyMissing(RuntimeError):
    """Ключ Gemini не задан в окружении."""


def api_key() -> str | None:
    """Ключ Gemini или None. Порядок переменных менять нельзя - см. модульный docstring."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def available() -> bool:
    return api_key() is not None


def require_api_key() -> str:
    key = api_key()
    if not key:
        raise GeminiKeyMissing("GOOGLE_API_KEY/GEMINI_API_KEY not set")
    return key


def _import_genai():
    """Импорт SDK с подавлением FutureWarning (шумит на каждом импорте)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

        return genai


def clinical_safety_settings() -> list[dict[str, Any]]:
    """Пороги безопасности для клинического текста.

    BLOCK_ONLY_HIGH по всем четырём категориям: дефолтный
    BLOCK_MEDIUM_AND_ABOVE блокирует обычные врачебные формулировки про травмы,
    самоповреждения и репродуктивное здоровье, и модель возвращает пустой ответ.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

    return [
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


def build_model(
    model_name: str,
    *,
    api_key_override: str | None = None,
    system_instruction: str | None = None,
):
    """Создаёт GenerativeModel с клиническими порогами безопасности.

    Единственный способ получить модель в проекте: любой вызов мимо этой
    функции рискует уехать на дефолтные пороги (см. модульный docstring).

    `api_key_override` нужен методистскому контуру в `rag_server.py`: там ключи
    перебираются по кругу при исчерпании квоты, поэтому ключ задаёт вызывающий,
    а не окружение.
    """
    genai = _import_genai()
    genai.configure(api_key=api_key_override or require_api_key())
    kwargs: dict[str, Any] = {"safety_settings": clinical_safety_settings()}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    return genai.GenerativeModel(model_name, **kwargs)
