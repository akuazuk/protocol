"""Калибровка уверенности выдачи протокола.

Объединяет доступные сигналы (опора отбора rag_support, релевантность по МКБ,
оценка модели, средняя опора извлечённых пунктов) в единую калиброванную вероятность
и относит её к понятной полосе (высокая/средняя/низкая). Детерминированно и монотонно
по каждому сигналу.

Примечание: полноценный нейросетевой cross-encoder-реранкер здесь намеренно не используется -
он требует существенно больше памяти/латентности, чем допускает текущий рантайм (Render 512 MiB).
Модуль даёт прозрачную, воспроизводимую калибровку на уже вычисленных признаках.
"""
from __future__ import annotations

import math
from typing import Any

# Веса сигналов (нормируются по фактически присутствующим).
_WEIGHTS: dict[str, float] = {
    "rag_support": 0.42,
    "icd_relevance": 0.30,
    "llm_confidence": 0.18,
    "grounding_avg": 0.10,
}

# Крутизна логистического сглаживания вокруг 0.5.
_LOGISTIC_K = 3.2


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _as01(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f > 1.0:  # проценты вида 92 -> 0.92
        f = f / 100.0
    return _clamp01(f)


def calibrate_confidence(
    *,
    rag_support: Any = None,
    icd_relevance: Any = None,
    llm_confidence: Any = None,
    grounding_avg: Any = None,
) -> float:
    """Калиброванная уверенность [0..1] из доступных сигналов."""
    signals = {
        "rag_support": _as01(rag_support),
        "icd_relevance": _as01(icd_relevance),
        "llm_confidence": _as01(llm_confidence),
        "grounding_avg": _as01(grounding_avg),
    }
    present = {k: v for k, v in signals.items() if v is not None}
    if not present:
        return 0.0
    wsum = sum(_WEIGHTS[k] for k in present)
    if wsum <= 0:
        return 0.0
    base = sum(_WEIGHTS[k] * present[k] for k in present) / wsum
    # логистическое сглаживание: растягивает уверенность к краям, сохраняя монотонность
    cal = 1.0 / (1.0 + math.exp(-_LOGISTIC_K * (base - 0.5)))
    return round(_clamp01(cal), 4)


def confidence_band(conf: Any) -> str:
    """высокая | средняя | низкая по калиброванной уверенности."""
    c = _as01(conf) or 0.0
    if c >= 0.75:
        return "высокая"
    if c >= 0.5:
        return "средняя"
    return "низкая"
