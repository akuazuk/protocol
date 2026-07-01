"""S2: локальный роутер query -> специальность / ICD-глава.

Лёгкий sklearn-классификатор (HashingVectorizer + OvR LogReg), обучен
`ml/train` на анализах. Задача - убрать Gemini из горячего пути маршрутизации
(симптом-поиск без готового диагноза).

Безопасность: модуль только предсказывает и (в shadow-режиме) логирует.
Он НЕ меняет клинические решения. При отсутствии модели или ошибке -
возвращает None (полный fallback на прежний путь).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("protocol.query_router")

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL = _ROOT / "ml/registry/checkpoints/query_route_v1/model.joblib"

_BUNDLE: dict[str, Any] | None = None
_LOAD_FAILED = False


def _model_path() -> Path:
    override = os.environ.get("RAG_QUERY_ROUTER_MODEL")
    return Path(override) if override else _DEFAULT_MODEL


def _load_bundle() -> dict[str, Any] | None:
    """Ленивая загрузка joblib-бандла; кэшируется. None при отсутствии/ошибке."""
    global _BUNDLE, _LOAD_FAILED
    if _BUNDLE is not None:
        return _BUNDLE
    if _LOAD_FAILED:
        return None
    path = _model_path()
    if not path.is_file():
        _LOAD_FAILED = True
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "vectorizer" not in bundle:
            _LOAD_FAILED = True
            return None
        _BUNDLE = bundle
        return _BUNDLE
    except Exception as exc:  # pragma: no cover - защитный fallback
        logger.warning("query_router load failed: %s", exc)
        _LOAD_FAILED = True
        return None


def is_available() -> bool:
    return _load_bundle() is not None


def _predict_one(clf, X) -> tuple[str | None, float | None]:
    try:
        pred = clf.predict(X)[0]
    except Exception:
        return None, None
    conf: float | None = None
    try:
        proba = clf.predict_proba(X)[0]
        classes = list(getattr(clf, "classes_", []))
        if classes and pred in classes:
            conf = float(proba[classes.index(pred)])
        elif len(proba):
            conf = float(max(proba))
    except Exception:
        conf = None
    return str(pred), conf


def predict_route(text: str) -> dict[str, Any] | None:
    """Предсказать специальность и ICD-главу по тексту запроса.

    Возвращает {"specialty_slug", "specialty_conf", "icd_chapter", "icd_chapter_conf"}
    или None, если модель недоступна / ошибка.
    """
    bundle = _load_bundle()
    if bundle is None:
        return None
    q = (text or "").strip()
    if not q:
        return None
    try:
        vec = bundle["vectorizer"]
        X = vec.transform([q])
    except Exception as exc:  # pragma: no cover
        logger.warning("query_router transform failed: %s", exc)
        return None
    spec, spec_conf = (None, None)
    chap, chap_conf = (None, None)
    if bundle.get("specialty_clf") is not None:
        spec, spec_conf = _predict_one(bundle["specialty_clf"], X)
    if bundle.get("icd_chapter_clf") is not None:
        chap, chap_conf = _predict_one(bundle["icd_chapter_clf"], X)
    return {
        "specialty_slug": spec,
        "specialty_conf": spec_conf,
        "icd_chapter": chap,
        "icd_chapter_conf": chap_conf,
    }


def shadow_enabled() -> bool:
    return os.environ.get("RAG_QUERY_ROUTER_SHADOW", "0").strip().lower() in ("1", "true", "yes")


def log_shadow(text: str, *, gemini_used: bool, detected_codes: list[str] | None = None) -> dict | None:
    """Shadow-режим: предсказать и залогировать, НЕ влияя на ответ.

    Логирует расхождение локального роутера с фактическим путём (Gemini/эвристика).
    Возвращает предсказание (для тестов) или None.
    """
    if not shadow_enabled():
        return None
    pred = predict_route(text)
    if pred is None:
        return None
    detected = [str(c) for c in (detected_codes or []) if c]
    actual_chapter = detected[0][0].upper() if detected and detected[0] else None
    chapter_match = (
        pred.get("icd_chapter") == actual_chapter if actual_chapter else None
    )
    payload = {
        "router_specialty": pred.get("specialty_slug"),
        "router_specialty_conf": pred.get("specialty_conf"),
        "router_icd_chapter": pred.get("icd_chapter"),
        "router_icd_chapter_conf": pred.get("icd_chapter_conf"),
        "actual_icd_chapter": actual_chapter,
        "icd_chapter_match": chapter_match,
        "gemini_used": bool(gemini_used),
    }
    try:
        logger.info("query_router_shadow %s", json.dumps(payload, ensure_ascii=False))
    except Exception:  # pragma: no cover
        pass
    return payload
